import os
import time
import sqlite3
import datetime
import asyncio
import urllib.parse
from typing import Optional, List, Dict, Any
from email.utils import parsedate_to_datetime
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

# ==========================
# 基本設定
# ==========================
APP_NAME = "stock-backend-pro"
DATABASE_PATH = os.getenv("DATABASE_PATH", "stock_app.db")

PRICE_CACHE: Dict[str, Any] = {}
FUND_CACHE: Dict[str, Any] = {}

PRICE_CACHE_TTL = 600       # 10分鐘
FUND_CACHE_TTL = 86400      # 1天

# ==========================
# FastAPI 初始化
# ==========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title=APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# DB
# ==========================
def get_db():
    conn = sqlite3.connect(
        DATABASE_PATH,
        check_same_thread=False,
        timeout=10
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            symbol TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(username, symbol)
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ==========================
# Models
# ==========================
class AnalysisRequest(BaseModel):
    symbol: str
    principal: float
    strategy: str
    duration: str

class FavoriteReq(BaseModel):
    symbol: str

# ==========================
# 技術指標
# ==========================
def compute_rsi(close, window=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)

def compute_macd(close):
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=9, adjust=False).mean()
    return macd, sig, macd - sig

def enrich_indicators(df):
    df["rsi"] = compute_rsi(df["Close"])
    macd, sig, hist = compute_macd(df["Close"])
    df["macd"] = macd
    df["macd_sig"] = sig
    df["macd_hist"] = hist
    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma60"] = df["Close"].rolling(60).mean()
    df["ma120"] = df["Close"].rolling(120).mean()
    return df

# ==========================
# 抓價格（快取 + threadpool）
# ==========================
async def fetch_price_history(symbol: str):
    now = time.time()

    if symbol in PRICE_CACHE:
        df, ts = PRICE_CACHE[symbol]
        if now - ts < PRICE_CACHE_TTL:
            return df

    def _download():
        df = yf.download(symbol, period="1y", interval="1d", progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna(subset=["Close"])
        df.index = pd.to_datetime(df.index)
        return df

    df = await run_in_threadpool(_download)
    if not df.empty:
        df = enrich_indicators(df)
        PRICE_CACHE[symbol] = (df, now)

    return df

# ==========================
# 基本面（快取 + fast_info）
# ==========================
async def score_fundamental(symbol):
    now = time.time()

    if symbol in FUND_CACHE:
        data, ts = FUND_CACHE[symbol]
        if now - ts < FUND_CACHE_TTL:
            return data

    def _fetch():
        return yf.Ticker(symbol).fast_info or {}

    info = await run_in_threadpool(_fetch)

    score = 50
    if info.get("forwardPE") and info["forwardPE"] < 20:
        score += 10

    result = {
        "score": max(0, min(100, score)),
        "forward_pe": info.get("forwardPE")
    }

    FUND_CACHE[symbol] = (result, now)
    return result

# ==========================
# 技術評分
# ==========================
def score_technical(df):
    latest = df.iloc[-1]
    score = 50

    if latest["Close"] > latest["ma20"]:
        score += 10
    if latest["ma20"] > latest["ma60"]:
        score += 10
    if latest["macd"] > latest["macd_sig"]:
        score += 8
    if latest["rsi"] < 30:
        score += 6
    elif latest["rsi"] > 70:
        score -= 6

    return max(0, min(100, score))

# ==========================
# 合成分數（動態權重）
# ==========================
def composite_score(tech, fund, volatility):
    if volatility > 0.5:
        wt = 0.45
    else:
        wt = 0.30

    return int(round(
        wt * tech +
        0.30 * fund +
        0.25 * 50
    ))

# ==========================
# API
# ==========================
@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):

    symbol = request.symbol.strip().upper()
    df = await fetch_price_history(symbol)

    if df.empty:
        raise HTTPException(status_code=404, detail="找不到資料")

    tech_score = score_technical(df)
    fund_score = await score_fundamental(symbol)

    volatility = df["Close"].pct_change().std() * np.sqrt(252)

    overall = composite_score(
        tech_score,
        fund_score["score"],
        volatility
    )

    return {
        "symbol": symbol,
        "price": round(float(df["Close"].iloc[-1]), 2),
        "technical_score": tech_score,
        "fundamental_score": fund_score,
        "ai_score": overall
    }

@app.get("/api/favorites")
async def get_favorites():
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT symbol FROM favorites")
    rows = c.fetchall()
    conn.close()
    return [r["symbol"] for r in rows]

@app.post("/api/favorites")
async def add_favorite(req: FavoriteReq):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT OR IGNORE INTO favorites (username, symbol, created_at) VALUES (?, ?, ?)",
        ("guest", req.symbol.upper(), datetime.datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    return {"message": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
