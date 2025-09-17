"""
Daily Stock Recommender Agent
=============================

This module implements a simple stock‑picking agent that runs once a day,
evaluates a basket of stocks using technical and fundamental indicators,
and sends an email containing 3–5 curated recommendations.  The agent
relies on the `yfinance` package to fetch market data and uses the
Relative Strength Index (RSI) together with the price/earnings ratio
and sector information to rank candidates.  If you wish to change the
screening universe or the scoring rules, edit the `WATCHLIST` and
`score_stock` functions below.

Usage
-----
The agent is designed to be run from the command line or invoked via a
scheduler like cron.  Before running it you must install its
dependencies and configure your email credentials.

1. Install Python dependencies (preferably in a virtual environment):

   ```bash
   pip install --upgrade yfinance pandas
   ```

2. Create a configuration file `.env` in the same directory or
   define the following environment variables in your shell:

   - `EMAIL_USER` – your email address (e.g. Gmail address).
   - `EMAIL_PASS` – an **app password** or SMTP password.  Gmail
     requires an app password when two‑factor authentication is
     enabled.  See <https://support.google.com/accounts/answer/185833>.
   - `EMAIL_TO` – comma‑separated list of recipients.  Set this to
     your own address to receive the recommendations.

3. Schedule the script to run at 8 AM America/Denver time.  On a
   Unix‑like system you can add the following line to your crontab
   (`crontab -e`):

   ```cron
   # run at 8:00 AM Denver time (adjust TZ as needed)
   0 8 * * 1-5 TZ=America/Denver /usr/bin/env python3 /path/to/stock_recommender_agent.py
   ```

Alternatively, run the script manually for testing:

```bash
python stock_recommender_agent.py --dry-run
```

When invoked with `--dry-run` the script prints the email message to
stdout instead of sending it.
"""

import datetime as _dt
import os
import smtplib
import sys
from dataclasses import dataclass
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Dict, List, Optional, Tuple

import pandas as pd  # type: ignore
import yfinance as yf  # type: ignore


###############################################################################
# Configuration
###############################################################################

# Define a set of tickers to monitor.  Feel free to add or remove symbols
# based on your sector preferences.  These tickers span AI/technology,
# clean energy, healthcare and semiconductors as requested by the user.
WATCHLIST: List[str] = [
    "NVDA",  # Nvidia – AI/semiconductors
    "META",  # Meta Platforms – AI/social media
    "WMT",  # Walmart – consumer staples, supply chain AI
    "FSLR",  # First Solar – clean energy
    "NEE",  # NextEra Energy – clean energy utility
    "MRK",  # Merck & Co. – healthcare/pharma
    "JNJ",  # Johnson & Johnson – healthcare
    "ABVX",  # Abivax – biotech (small cap)
    "WOLF",  # Wolfspeed – semiconductors
    "AMD",  # Advanced Micro Devices – semiconductors/AI
    "PLTR",  # Palantir Technologies – AI/software
]

# Screening parameters.  The agent will flag a stock as oversold when
# RSI falls below this threshold and as undervalued when the trailing
# price/earnings ratio (PE) or forward PE is below this value.  Market
# capitalization filter helps avoid extremely thinly traded microcaps.
RSI_THRESHOLD: float = 30.0
PE_THRESHOLD: float = 25.0
MIN_MARKET_CAP: float = 5e9  # $5 Billion

# Number of recommendations to include in the daily report.  This
# parameter is used when more than the desired number of stocks meet
# the screening criteria.  The highest scoring names will be picked.
MAX_RECOMMENDATIONS: int = 5

# Email settings.  These values are read from environment variables
# (recommended) or fallback to the constants below if set.  Use a
# `.env` file or export variables in your shell for security.
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))  # SSL port


###############################################################################
# Data classes and utility functions
###############################################################################

@dataclass
class StockMetrics:
    """Container for computed metrics used in scoring stocks."""

    ticker: str
    current_price: float
    trailing_pe: Optional[float]
    forward_pe: Optional[float]
    market_cap: Optional[float]
    sector: Optional[str]
    rsi: Optional[float]
    # Additional fields can be added (e.g., beta, sentiment score).


def compute_rsi(prices: pd.Series, window: int = 14) -> Optional[float]:
    """Return the most recent Relative Strength Index (RSI).

    The RSI is a momentum indicator that measures the magnitude of recent
    price changes.  It oscillates between 0 and 100, with values below
    `RSI_THRESHOLD` typically interpreted as oversold and values above
    70 as overbought.  If fewer than `window + 1` data points are
    available, ``None`` is returned.

    Parameters
    ----------
    prices : pd.Series
        Series of closing prices ordered from most recent to oldest.
    window : int, optional
        The number of periods to use for the RSI calculation, by
        default 14.

    Returns
    -------
    float or None
        The most recent RSI value or ``None`` if not enough data.
    """
    if len(prices) < window + 1:
        return None
    # Compute differences between consecutive prices
    delta = prices.diff().iloc[1:]
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window).mean().iloc[-1]
    avg_loss = losses.rolling(window=window).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0  # avoid division by zero (no losses means RSI=100)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi)


def fetch_metrics(ticker: str) -> Optional[StockMetrics]:
    """Fetch pricing and fundamental information for a given stock.

    This function uses `yfinance` to download recent price history and
    basic fundamental data.  It returns a `StockMetrics` object on
    success or ``None`` if data cannot be retrieved.

    Parameters
    ----------
    ticker : str
        The stock symbol to query.

    Returns
    -------
    StockMetrics or None
        Container with computed metrics or ``None`` if data is missing.
    """
    try:
        yticker = yf.Ticker(ticker)
        info: Dict[str, Optional[float]] = yticker.info  # type: ignore
        # Some tickers return an empty dict when the data cannot be fetched
        if not info:
            return None
        # Fetch closing prices for the last 30 trading days
        end = _dt.datetime.now()
        start = end - _dt.timedelta(days=60)
        history = yticker.history(start=start.date(), end=end.date(), interval="1d")
        close_prices = history["Close"].dropna()
        rsi_value = compute_rsi(close_prices.reset_index(drop=True), window=14)
        return StockMetrics(
            ticker=ticker,
            current_price=float(close_prices.iloc[-1]) if not close_prices.empty else float("nan"),
            trailing_pe=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            market_cap=info.get("marketCap"),
            sector=info.get("sector"),
            rsi=rsi_value,
        )
    except Exception as exc:
        # Print error to stderr for debugging; return None to skip this ticker
        print(f"Error fetching data for {ticker}: {exc}", file=sys.stderr)
        return None


def score_stock(metrics: StockMetrics) -> Tuple[float, List[str]]:
    """Assign a numerical score to a stock based on screening rules.

    A higher score indicates a more attractive investment candidate.  The
    rules implemented here favour stocks that are oversold (RSI <
    ``RSI_THRESHOLD``), have low P/E ratios (below ``PE_THRESHOLD``), and
    possess substantial market caps.  Each rule contributes points to
    the total score.  The function also returns a list of textual
    justifications used later when constructing the report.

    Parameters
    ----------
    metrics : StockMetrics
        The computed metrics for a stock.

    Returns
    -------
    (float, List[str])
        Tuple containing the total score and a list of reasons.
    """
    score = 0.0
    reasons: List[str] = []
    # RSI rule: oversold condition adds points
    if metrics.rsi is not None:
        if metrics.rsi < RSI_THRESHOLD:
            score += 2.0
            reasons.append(f"RSI {metrics.rsi:.1f} (<{RSI_THRESHOLD}) – oversold")
        elif metrics.rsi < 50:
            score += 0.5  # mild positive
            reasons.append(f"RSI {metrics.rsi:.1f} (<50) – moderate momentum")
        else:
            reasons.append(f"RSI {metrics.rsi:.1f} – neutral/overbought")
    else:
        reasons.append("RSI unavailable")
    # PE rules: undervalued or fairly valued
    pe = metrics.trailing_pe or metrics.forward_pe
    if pe is not None:
        if pe < PE_THRESHOLD:
            score += 1.5
            reasons.append(f"P/E {pe:.1f} (<{PE_THRESHOLD}) – undervalued")
        elif pe < 40:
            score += 0.5
            reasons.append(f"P/E {pe:.1f} – fair value")
        else:
            reasons.append(f"P/E {pe:.1f} – high valuation")
    else:
        reasons.append("P/E unavailable")
    # Market cap: bigger companies earn additional confidence points
    if metrics.market_cap and metrics.market_cap >= MIN_MARKET_CAP:
        score += 0.5
        reasons.append(f"Market cap {metrics.market_cap/1e9:.1f}B ≥ {MIN_MARKET_CAP/1e9:.1f}B")
    elif metrics.market_cap:
        reasons.append(f"Market cap {metrics.market_cap/1e9:.1f}B < {MIN_MARKET_CAP/1e9:.1f}B")
    else:
        reasons.append("Market cap unavailable")
    return score, reasons


def prepare_recommendations(metrics_list: List[StockMetrics]) -> List[Tuple[StockMetrics, float, List[str]]]:
    """Filter and rank stocks based on their computed scores.

    The function applies the scoring rules via `score_stock`, sorts the
    stocks by descending score, and selects the top `MAX_RECOMMENDATIONS`.
    Stocks with non‑finite or undefined scores are automatically
    excluded.  Returns a list of tuples containing the metrics, score and
    reasons for each recommended stock.
    """
    scored: List[Tuple[StockMetrics, float, List[str]]] = []
    for metrics in metrics_list:
        # Skip if required fields are missing
        if metrics.market_cap is None or (metrics.trailing_pe is None and metrics.forward_pe is None):
            continue
        score, reasons = score_stock(metrics)
        scored.append((metrics, score, reasons))
    # Sort by score descending
    scored.sort(key=lambda tup: tup[1], reverse=True)
    # Take the top N
    return scored[:MAX_RECOMMENDATIONS]


def build_email_content(recommendations: List[Tuple[StockMetrics, float, List[str]]], report_date: _dt.date) -> str:
    """Construct a formatted email body in Markdown.

    Each recommendation includes the ticker symbol, current price,
    relevant metrics (PE, RSI, sector), and bullet points explaining
    why the stock was selected.  A summary header includes the date
    and a disclaimer.  The returned string can be sent as plain text
    or HTML (most email clients render Markdown gracefully).
    """
    lines: List[str] = []
    lines.append(f"**Daily Stock Recommendations – {report_date.strftime('%B %d, %Y')}**\n")
    if not recommendations:
        lines.append("No stocks met the screening criteria today.\n")
    else:
        for metrics, score, reasons in recommendations:
            pe_value = metrics.trailing_pe or metrics.forward_pe
            pe_str = f"{pe_value:.1f}" if pe_value is not None else "n/a"
            rsi_str = f"{metrics.rsi:.1f}" if metrics.rsi is not None else "n/a"
            sector = metrics.sector or "N/A"
            lines.append(f"### {metrics.ticker} – Price: ${metrics.current_price:.2f}\n")
            lines.append(f"Sector: {sector} | P/E: {pe_str} | RSI: {rsi_str}\n")
            # Provide a recommendation label
            label = "Buy" if score >= 2.5 else "Hold" if score >= 1.0 else "Watch"
            lines.append(f"**Recommendation:** {label}\n")
            for reason in reasons:
                lines.append(f"- {reason}\n")
            lines.append("\n")
    # Add a brief disclaimer
    lines.append("*This report is generated automatically for informational purposes only and does not constitute financial advice.*\n")
    return "\n".join(lines)


def send_email(subject: str, body: str, sender: str, password: str, recipients: List[str], smtp_server: str = SMTP_SERVER, smtp_port: int = SMTP_PORT) -> None:
    """Send an email via an SMTP server using SSL.

    Parameters
    ----------
    subject : str
        Subject line for the email.
    body : str
        Plain‑text or HTML body of the message.  Markdown is acceptable for
        many mail clients.
    sender : str
        Email address to send from.
    password : str
        SMTP password or app password for the sending account.
    recipients : List[str]
        List of recipient email addresses.
    smtp_server : str, optional
        Hostname of the SMTP server, by default configured from
        environment (gmail).
    smtp_port : int, optional
        Port for the SMTP server, by default configured for SSL.
    """
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(sender, password)
        server.sendmail(sender, recipients, msg.as_string())


def load_env_file(filepath: str = ".env") -> None:
    """Load environment variables from a `.env` file if it exists.

    Variables defined in the existing environment are not overwritten.
    """
    try:
        with open(filepath) as f:
            for line in f:
                if not line.strip() or line.strip().startswith("#"):
                    continue
                key, _, value = line.strip().partition("=")
                if key and value and key not in os.environ:
                    os.environ[key] = value
    except FileNotFoundError:
        pass


def main(dry_run: bool = False) -> None:
    """Entry point for the stock recommender.

    When `dry_run` is True the email is not sent; instead the report is
    printed to stdout.  This is useful for testing.  On a normal run
    the function reads environment variables for email credentials,
    fetches data for all tickers in the watchlist, computes
    recommendations, builds the email message, and sends it.
    """
    # Load environment variables from .env file if available
    load_env_file()
    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")
    recipients_raw = os.getenv("EMAIL_TO")
    if not sender or not password or not recipients_raw:
        raise RuntimeError(
            "EMAIL_USER, EMAIL_PASS, and EMAIL_TO must be set in the environment or .env file."
        )
    recipients = [email.strip() for email in recipients_raw.split(",") if email.strip()]
    # Fetch metrics for all tickers
    metrics_list: List[StockMetrics] = []
    for ticker in WATCHLIST:
        metrics = fetch_metrics(ticker)
        if metrics is not None:
            metrics_list.append(metrics)
    # Prepare recommendations
    recommendations = prepare_recommendations(metrics_list)
    report_date = _dt.date.today()
    body = build_email_content(recommendations, report_date)
    subject = f"Daily Stock Recommendations – {report_date.isoformat()}"
    if dry_run:
        print(body)
    else:
        send_email(subject, body, sender, password, recipients)


if __name__ == "__main__":
    # Support a --dry-run command line option
    dry_run_flag = "--dry-run" in sys.argv
    main(dry_run=dry_run_flag)