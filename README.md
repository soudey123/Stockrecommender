# Stock Recommendation Agent 📈

AI-powered stock screening system that combines classical analysis with GPT-5 reasoning to rank stocks and email recommendations automatically.

## Features

- **Triple Engine**: Classical momentum + GPT-5 reasoning + hybrid scoring
- **Live Data**: Real-time market data via yfinance
- **Auto Email**: One-click delivery of top picks
- **Interactive UI**: Streamlit interface with adjustable parameters

## Quick Start

```bash
# Clone and setup
git clone https://github.com/yourusername/stock-recommendation-agent.git
cd stock-recommendation-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ./Stockrecommender

# Configure secrets
export OPENAI_API_KEY="sk-your-key"
export EMAIL_USER="your@gmail.com"
export EMAIL_PASS="gmail-app-password"
export EMAIL_TO="recipient@domain.com"

# Run
streamlit run Stockrecommender/main_gpt.py
```

## How It Works

1. **Data**: Fetches OHLCV data and calculates returns, volatility, volume
2. **Scoring**: 
   - Classical: momentum-based ranking (0-1)
   - GPT-5: structured reasoning with JSON output
   - Hybrid: weighted blend of both engines
3. **Action**: Emails top-K picks as formatted HTML

## Usage

1. Enter tickers: `AAPL,MSFT,GOOGL,TSLA`
2. Select engine and adjust parameters
3. Review recommendations table
4. Click "Send Email" for top picks

Sample output:
```
Ticker  Score  
GOOGL   1.000  
AAPL    0.427  
TSLA    0.307  
```

## Configuration

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | GPT-5 access |
| `EMAIL_USER` | Gmail sender |
| `EMAIL_PASS` | Gmail app password |
| `EMAIL_TO` | Default recipient |

**Gmail Setup**: Enable 2FA → Generate app password → Use in `EMAIL_PASS`

## Architecture

```
Data (yfinance) → Scoring (Classical/GPT-5/Hybrid) → Email (SMTP)
```

Core files:
- `main_gpt.py` - Streamlit UI
- `gpt_recommender.py` - GPT-5 integration  
- `email_utils.py` - SMTP functionality

## ⚠️ Disclaimer

Educational purposes only. Not financial advice. Consult qualified professionals before investing.

---

**Star ⭐ if helpful!**