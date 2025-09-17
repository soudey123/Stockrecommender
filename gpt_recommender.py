from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import OpenAI
import os

class TickerSnapshot(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g., AAPL")
    price: float
    avg_vol_20d: float
    ret_5d: float
    ret_20d: float
    vol_20d: float
    rsi_14: float | None = None
    macd: float | None = None
    sector: str | None = None

class TickerScore(BaseModel):
    ticker: str
    score: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str
    risk_flags: List[str] = []

class RerankResponse(BaseModel):
    picks: List[TickerScore]

_SYSTEM_PROMPT = """You are a disciplined equity ranking model.
Given compact, numeric snapshots per ticker, return a JSON object that scores each ticker between 0 and 1 for the next H trading days.
Prioritize: momentum persistence, liquidity, volatility control, and sector balance.
Do NOT extrapolate beyond provided numbers. Do NOT give financial advice. Be concise but specific in rationales."""

def _client() -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=key)

def rerank_with_gpt5(snapshots: List[TickerSnapshot], horizon_days: int, max_picks: int) -> RerankResponse:
    client = _client()

    payload = {
        "horizon_days": int(horizon_days),
        "max_picks": int(max_picks),
        "snapshots": [s.model_dump() for s in snapshots],
    }

    # JSON schema for strict output validation
    schema = {
        "name": "RerankResponse",
        "schema": {
            "type": "object",
            "properties": {
                "picks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": "string"},
                            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "rationale": {"type": "string"},
                            "risk_flags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["ticker", "score", "confidence", "rationale"]
                    }
                }
            },
            "required": ["picks"]
        }
    }

    resp = client.responses.create(
        model="gpt-5",
        temperature=0.2,
        reasoning={"effort": "medium"},
        system=_SYSTEM_PROMPT,
        response_format={"type": "json_schema", "json_schema": schema},
        input=[{"role": "user", "content": str(payload)}],
    )

    # Extract JSON string content (OpenAI Python SDK v1+)
    content = resp.output[0].content[0].text
    # Pydantic parse
    return RerankResponse.model_validate_json(content)

def blend_scores(numeric_scores: Dict[str, float], llm_scores: Dict[str, float], w: float) -> Dict[str, float]:
    w = max(0.0, min(1.0, float(w)))
    keys = set(numeric_scores) | set(llm_scores)
    return {k: (1 - w) * numeric_scores.get(k, 0.0) + w * llm_scores.get(k, 0.0) for k in keys}
