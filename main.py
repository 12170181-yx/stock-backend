import os
import time
import datetime
import asyncio
import numpy as np
import pandas as pd
import pandas_ta as ta  # 🚀 引入最強大的技術指標庫
import yfinance as yf
import aiosqlite

from typing import Dict, Any, Tuple
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

APP_NAME = "stock-backend-quant-pro"
DATABASE_PATH = os.getenv("DATABASE_PATH", "stock_app.db")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

PRICE_CACHE: Dict[str, Tuple[pd.DataFrame, float]] = {}
FUND_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}

PRICE_CACHE_TTL = 600
FUND_CACHE_TTL = 86400

# ==========================
# FastAPI 初始化 & DB
# ==========================
async def init_db():
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("""
            CREATE TABLE IF NOT EXISTS favorites (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                symbol TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(username, symbol)
            )
        """)
        await db.commit()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(title=APP_NAME, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    symbol: str
    principal: float
    duration: str

# ==========================
# 📊 核心：海量指標計算引擎 (基於 pandas-ta)
# ==========================
def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """使用 pandas-ta 快速擴充所有技術指標"""
    # 計算日報酬率
    df["ret"] = df["Close"].pct_change()

    # 一、均線系統 (Moving Averages)
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.vwma(length=20, append=True)
    df.ta.hma(length=20, append=True) # Hull MA
    df.ta.kama(length=10, append=True) # Kaufman Adaptive MA
    
    # 二、動能與趨勢類 (Momentum & Trend)
    df.ta.macd(append=True)
    df.ta.adx(append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(append=True) # KD (Stochastic)
    df.ta.cci(length=20, append=True)
    df.ta.willr(append=True) # Williams %R
    df.ta.roc(length=10, append=True) # Rate of Change
    
    # 三、波動率類 (Volatility)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.kc(append=True) # Keltner Channel
    df.ta.donchian(append=True) # Donchian Channel
    
    # 四、成交量類 (Volume)
    df.ta.obv(append=True)
    df.ta.mfi(length=14, append=True) # Money Flow Index
    df.ta.vwap(append=True)
    df.ta.cmf(append=True) # Chaikin Money Flow
    
    # 五、型態辨識 (Candlestick Patterns)
    # pandas-ta 支援自動辨識吞噬、十字星等，這裡算出一個綜合訊號 (大於0偏多，小於0偏空)
    df.ta.cdl_pattern(name=["doji", "engulfing", "hammer"], append=True)
    
    # 六、統計量化指標 (Quant)
    df.ta.zscore(length=20, append=True) # Z-Score
    df.ta.variance(length=20, append=True)
    
    # 計算波動度 (作為動態權重切換依據)
    df["volatility_20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    
    # 清理 NaN (因為許多長天期指標會有空值)
    return df.dropna(subset=["SMA_20", "volatility_20", "MACD_12_26_9"])

# ==========================
# 📡 抓取資料模組
# ==========================
async def fetch_price_history(symbol: str) -> pd.DataFrame:
    now = time.time()
    if symbol in PRICE_CACHE:
        df, ts = PRICE_CACHE[symbol]
        if now - ts < PRICE_CACHE_TTL:
            return df

    def _download():
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.dropna(subset=["Close", "Volume"])
        df.index = pd.to_datetime(df.index)
        return enrich_indicators(df)

    df = await run_in_threadpool(_download)
    if not df.empty:
        PRICE_CACHE[symbol] = (df, now)
    return df

async def fetch_fundamental(symbol: str) -> Dict[str, float]:
    now = time.time()
    if symbol in FUND_CACHE:
        data, ts = FUND_CACHE[symbol]
        if now - ts < FUND_CACHE_TTL:
            return data

    def _fetch():
        info = yf.Ticker(symbol).fast_info
        return {"pe_ratio": getattr(info, "trailingPE", 15) or 15}

    try:
        data = await run_in_threadpool(_fetch)
    except Exception:
        data = {"pe_ratio": 15}
        
    FUND_CACHE[symbol] = (data, now)
    return data

# ==========================
# 🧠 AI 綜合評分系統 (匯總海量指標)
# ==========================
def calculate_multi_factor_score(df: pd.DataFrame, fund_data: Dict[str, float]) -> Dict[str, Any]:
    """將七大類指標壓縮成四個維度的分數，並根據 Regime 給予權重"""
    latest = df.iloc[-1]
    
    # 1. 趨勢分數 (Trend Score) - 綜合 MA, MACD, ADX
    trend_score = 50
    if latest.get("Close") > latest.get("SMA_20", 0): trend_score += 15
    if latest.get("MACD_12_26_9", 0) > latest.get("MACDs_12_26_9", 0): trend_score += 15
    if latest.get("ADX_14", 0) > 25: trend_score += 10 # 趨勢確立
    if latest.get("Close") > latest.get("VWMA_20", 0): trend_score += 10
    
    # 2. 動能分數 (Momentum Score) - 綜合 RSI, KD, CCI
    mom_score = 50
    rsi = latest.get("RSI_14", 50)
    if 40 < rsi < 70: mom_score += 20
    elif rsi >= 70: mom_score -= 10 # 超買
    elif rsi <= 30: mom_score += 10 # 超賣反彈預期
    
    cci = latest.get("CCI_20_0.015", 0)
    if cci > 100: mom_score += 10
    elif cci < -100: mom_score -= 10
    
    # 3. 籌碼/量能分數 (Volume Score) - 綜合 OBV, MFI, CMF
    vol_score = 50
    if latest.get("MFI_14", 50) > 50: vol_score += 25
    if latest.get("CMF_20", 0) > 0: vol_score += 25
    
    # 4. 基本面分數 (Fundamental)
    fund_score = 100 - (min(fund_data["pe_ratio"], 50) * 2)
    fund_score = max(0, min(100, fund_score))
    
    # --- Regime Switching 動態權重 ---
    is_high_vol = latest.get("volatility_20", 0) > 0.35
    
    if is_high_vol:
        regime = "高波動防禦期"
        # 高波動時，趨勢與動能容易失效，看重量能與基本面
        final_score = (trend_score * 0.1) + (mom_score * 0.1) + (vol_score * 0.4) + (fund_score * 0.4)
    else:
        regime = "穩健趨勢期"
        # 低波動時，順勢交易為主
        final_score = (trend_score * 0.4) + (mom_score * 0.3) + (vol_score * 0.15) + (fund_score * 0.15)

    return {
        "final_score": int(max(0, min(100, final_score))),
        "breakdown": {
            "technical": int((trend_score + mom_score)/2),
            "fundamental": int(fund_score),
            "chip": int(vol_score),
            "news": 60 # 外部新聞預設
        },
        "regime": regime
    }

# ==========================
# 📊 API 路由
# ==========================
@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    symbol = request.symbol.strip().upper()
    df = await fetch_price_history(symbol)

    if df.empty:
        raise HTTPException(status_code=404, detail="找不到該股票資料")

    fund_data = await fetch_fundamental(symbol)
    scoring = calculate_multi_factor_score(df, fund_data)
    
    # 計算 CVaR (停損參考)
    recent_rets = df["ret"].tail(60).dropna()
    var_95 = recent_rets.quantile(0.05)
    cvar_95 = recent_rets[recent_rets <= var_95].mean() * 100 if not recent_rets.empty else -5

    last_price = float(df["Close"].iloc[-1])
    
    # 模擬預測 (配合前端)
    days = 14 if request.duration == "short" else 60 if request.duration == "mid" else 180
    drift = recent_rets.mean()
    vol = df["volatility_20"].iloc[-1] / np.sqrt(252)
    
    predictions = []
    sim_p = last_price
    for i in range(1, days + 1):
        sim_p *= (1 + np.random.normal(drift, vol))
        predictions.append({
            "date": (datetime.datetime.now() + datetime.timedelta(days=i)).strftime("%m-%d"),
            "mid": round(sim_p, 2)
        })

    history_data = [
        {"date": idx.strftime("%m-%d"), "price": round(float(row["Close"]), 2)}
        for idx, row in df.tail(30).iterrows()
    ]

    return {
        "symbol": symbol,
        "ai_score": scoring["final_score"],
        "ai_sentiment": scoring["regime"],
        "score_breakdown": scoring["breakdown"],
        "advice": {
            "buy_price": round(last_price * 0.98, 2),
            "take_profit": predictions[-1]["mid"],
            "stop_loss": round(last_price * (1 + (cvar_95/100)), 2)
        },
        "chart_data": {
            "history": history_data,
            "prediction": predictions
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
