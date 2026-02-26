import os
import time
import datetime
import asyncio
import urllib.parse
import numpy as np
import pandas as pd
import pandas_ta as ta  # 🚀 最強大的技術指標庫
import yfinance as yf
import aiosqlite
import feedparser       # 新聞 RSS 解析

from typing import Dict, Any, Tuple, List, Optional
from email.utils import parsedate_to_datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

APP_NAME = "stock-backend-quant-pro"
DATABASE_PATH = os.getenv("DATABASE_PATH", "stock_app.db")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

# ==========================
# 🗄️ 全域快取系統 (Cache)
# ==========================
PRICE_CACHE: Dict[str, Tuple[pd.DataFrame, float]] = {}
FUND_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}
NEWS_CACHE: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}

PRICE_CACHE_TTL = 600      # 股價快取 10 分鐘
FUND_CACHE_TTL = 86400     # 基本面快取 1 天
NEWS_CACHE_TTL = 3600      # 新聞快取 1 小時

# ==========================
# ⚙️ FastAPI 初始化 & DB
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
# 📰 外部新聞抓取模組 (非同步 + 快取)
# ==========================
def parse_dt(entry: dict) -> Optional[datetime.datetime]:
    if entry.get("published_parsed"):
        try: return datetime.datetime(*entry["published_parsed"][:6])
        except: pass
    if entry.get("published"):
        try: return parsedate_to_datetime(entry["published"])
        except: return None
    return None

def _fetch_rss_sync(query: str, limit: int) -> List[Dict[str, Any]]:
    """在背景執行緒跑的同步抓取邏輯"""
    safe_query = urllib.parse.quote(query.strip())
    rss_url = f"https://news.google.com/rss/search?q={safe_query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    
    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        print(f"RSS Fetch Error: {e}")
        return []

    items = []
    for e in feed.entries[:limit * 2]:
        dt = parse_dt(e)
        title = getattr(e, "title", "")
        
        # 簡易 NLP 標籤分類
        tag = "產業"
        low_title = title.lower()
        if any(k in low_title for k in ["風險", "戰", "通膨", "升息", "fed", "跌", "警告"]): tag = "風險"
        if any(k in low_title for k in ["法說", "財報", "營收", "目標價", "漲", "創新高"]): tag = "評論"
        
        items.append({
            "tag": tag,
            "time": getattr(e, "published", "") or (dt.strftime("%Y-%m-%d %H:%M") if dt else ""),
            "published_at": dt.isoformat() if dt else "",
            "title": title,
            "url": getattr(e, "link", ""),
            "source": getattr(e, "source", {}).get("title", "Google News") if hasattr(e, "source") else "Google News"
        })
        
    # 去重與排序
    seen = set()
    uniq = []
    for it in items:
        if it["url"] not in seen:
            seen.add(it["url"])
            uniq.append(it)
            
    uniq.sort(key=lambda x: x["published_at"], reverse=True)
    return uniq[:limit]

@app.get("/api/news")
async def get_news(q: str = Query("全球市場 財經"), limit: int = Query(10, ge=1, le=20)):
    now = time.time()
    cache_key = f"{q}_{limit}"
    
    # 1. 檢查快取
    if cache_key in NEWS_CACHE:
        data, ts = NEWS_CACHE[cache_key]
        if now - ts < NEWS_CACHE_TTL:
            return data
            
    # 2. 快取失效，將 blocking IO 丟入 threadpool
    data = await run_in_threadpool(_fetch_rss_sync, q, limit)
    
    # 3. 更新快取
    if data:
        NEWS_CACHE[cache_key] = (data, now)
        
    return data

# ==========================
# 📊 核心：海量指標計算引擎 (pandas-ta)
# ==========================
def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ret"] = df["Close"].pct_change()

    # 一、均線系統 
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.vwma(length=20, append=True)
    
    # 二、動能與趨勢類
    df.ta.macd(append=True)
    df.ta.adx(append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.cci(length=20, append=True)
    
    # 三、波動率類
    df.ta.bbands(length=20, std=2, append=True)
    
    # 四、成交量類 
    df.ta.mfi(length=14, append=True) 
    df.ta.cmf(append=True) 
    
    # 計算日波動度
    df["volatility_20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    
    return df.dropna(subset=["SMA_20", "volatility_20", "MACD_12_26_9"])

# ==========================
# 📡 抓取金融資料模組
# ==========================
async def fetch_price_history(symbol: str) -> pd.DataFrame:
    now = time.time()
    if symbol in PRICE_CACHE:
        df, ts = PRICE_CACHE[symbol]
        if now - ts < PRICE_CACHE_TTL:
            return df

    def _download():
        df = yf.download(symbol, period="2y", interval="1d", progress=False)
        if df is None or df.empty: return pd.DataFrame()
        df = df.dropna(subset=["Close", "Volume"])
        df.index = pd.to_datetime(df.index)
        return enrich_indicators(df)

    df = await run_in_threadpool(_download)
    if not df.empty: PRICE_CACHE[symbol] = (df, now)
    return df

async def fetch_fundamental(symbol: str) -> Dict[str, float]:
    now = time.time()
    if symbol in FUND_CACHE:
        data, ts = FUND_CACHE[symbol]
        if now - ts < FUND_CACHE_TTL: return data

    def _fetch():
        info = yf.Ticker(symbol).fast_info
        return {"pe_ratio": getattr(info, "trailingPE", 15) or 15}

    try: data = await run_in_threadpool(_fetch)
    except Exception: data = {"pe_ratio": 15}
        
    FUND_CACHE[symbol] = (data, now)
    return data

# ==========================
# 🧠 AI 綜合多因子評分系統
# ==========================
def calculate_multi_factor_score(df: pd.DataFrame, fund_data: Dict[str, float]) -> Dict[str, Any]:
    latest = df.iloc[-1]
    
    # 1. 趨勢分數
    trend_score = 50
    if latest.get("Close") > latest.get("SMA_20", 0): trend_score += 15
    if latest.get("MACD_12_26_9", 0) > latest.get("MACDs_12_26_9", 0): trend_score += 15
    if latest.get("ADX_14", 0) > 25: trend_score += 10 
    
    # 2. 動能分數
    mom_score = 50
    rsi = latest.get("RSI_14", 50)
    if 40 < rsi < 70: mom_score += 20
    elif rsi >= 70: mom_score -= 10 
    elif rsi <= 30: mom_score += 10 
    cci = latest.get("CCI_20_0.015", 0)
    if cci > 100: mom_score += 10
    elif cci < -100: mom_score -= 10
    
    # 3. 量能分數 
    vol_score = 50
    if latest.get("MFI_14", 50) > 50: vol_score += 25
    if latest.get("CMF_20", 0) > 0: vol_score += 25
    
    # 4. 基本面分數
    fund_score = max(0, min(100, 100 - (min(fund_data["pe_ratio"], 50) * 2)))
    
    # --- 動態體制切換 (Regime Switching) ---
    is_high_vol = latest.get("volatility_20", 0) > 0.35
    
    if is_high_vol:
        regime = "高波動防禦期"
        final_score = (trend_score * 0.1) + (mom_score * 0.1) + (vol_score * 0.4) + (fund_score * 0.4)
    else:
        regime = "穩健趨勢期"
        final_score = (trend_score * 0.4) + (mom_score * 0.3) + (vol_score * 0.15) + (fund_score * 0.15)

    return {
        "final_score": int(max(0, min(100, final_score))),
        "breakdown": {
            "technical": int((trend_score + mom_score)/2),
            "fundamental": int(fund_score),
            "chip": int(vol_score),
            "news": 65 # 結合前端顯示預設分數
        },
        "regime": regime
    }

# ==========================
# 🚀 股票分析 API 路由
# ==========================
@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    symbol = request.symbol.strip().upper()
    df = await fetch_price_history(symbol)

    if df.empty:
        raise HTTPException(status_code=404, detail="找不到該股票資料")

    fund_data = await fetch_fundamental(symbol)
    scoring = calculate_multi_factor_score(df, fund_data)
    
    # CVaR (預期落差) 計算
    recent_rets = df["ret"].tail(60).dropna()
    var_95 = recent_rets.quantile(0.05)
    cvar_95 = recent_rets[recent_rets <= var_95].mean() * 100 if not recent_rets.empty else -5

    last_price = float(df["Close"].iloc[-1])
    
    # 隨機漫步模擬預測
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
