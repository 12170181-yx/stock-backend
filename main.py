import os
import time
import datetime
import asyncio
import urllib.parse
import numpy as np
import pandas as pd
import pandas_ta as ta  # 🚀 最強大的技術指標庫
import aiosqlite
import feedparser       # 新聞 RSS 解析
import requests         # API 請求套件

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

# 🚀 讀取你在 Render 設定的 FMP API Key
FMP_API_KEY = os.getenv("FMP_API_KEY", "demo")

# ==========================
# 🗄️ 全域快取系統 (Cache)
# ==========================
PRICE_CACHE: Dict[str, Tuple[pd.DataFrame, float]] = {}
FUND_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}
NEWS_CACHE: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}

PRICE_CACHE_TTL = 600      
FUND_CACHE_TTL = 86400     
NEWS_CACHE_TTL = 3600      

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
    CORSMiddleware, 
    allow_origins=["*"],  
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"],
)

class AnalysisRequest(BaseModel):
    symbol: str
    principal: float
    duration: str

# ==========================
# 📰 外部新聞抓取模組
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
    safe_query = urllib.parse.quote(query.strip())
    rss_url = f"https://news.google.com/rss/search?q={safe_query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    try: feed = feedparser.parse(rss_url)
    except Exception: return []

    items = []
    for e in feed.entries[:limit * 2]:
        dt = parse_dt(e)
        title = getattr(e, "title", "")
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
    if cache_key in NEWS_CACHE:
        data, ts = NEWS_CACHE[cache_key]
        if now - ts < NEWS_CACHE_TTL: return data
    data = await run_in_threadpool(_fetch_rss_sync, q, limit)
    if data: NEWS_CACHE[cache_key] = (data, now)
    return data

# ==========================
# 📊 核心：海量指標計算引擎
# ==========================
def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ret"] = df["Close"].pct_change()
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.vwma(length=20, append=True)
    df.ta.macd(append=True)
    df.ta.adx(append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.cci(length=20, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.mfi(length=14, append=True) 
    df.ta.cmf(append=True) 
    df["volatility_20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    return df.dropna(subset=["SMA_20", "volatility_20", "MACD_12_26_9"])

# ==========================
# 🔀 智慧路由：判斷台股或美股
# ==========================
def is_taiwan_stock(symbol: str) -> bool:
    # 如果代碼是純數字，或者帶有 .TW / .TWO，判定為台股
    clean_symbol = symbol.replace(".TW", "").replace(".TWO", "")
    return clean_symbol.isdigit()

def clean_tw_symbol(symbol: str) -> str:
    # 將 2330.TW 轉為純數字 2330 供 FinMind 查詢
    return symbol.replace(".TW", "").replace(".TWO", "")

# ==========================
# 📡 抓取金融資料模組 (台美雙引擎)
# ==========================
async def fetch_price_history(symbol: str) -> pd.DataFrame:
    now = time.time()
    if symbol in PRICE_CACHE:
        df, ts = PRICE_CACHE[symbol]
        if now - ts < PRICE_CACHE_TTL: return df

    def _download():
        if is_taiwan_stock(symbol):
            # 🇹🇼 走 FinMind 通道 (台股)
            tw_id = clean_tw_symbol(symbol)
            start_date = (datetime.datetime.now() - datetime.timedelta(days=700)).strftime("%Y-%m-%d")
            url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id={tw_id}&start_date={start_date}"
            resp = requests.get(url)
            if resp.status_code != 200: return pd.DataFrame()
            data = resp.json()
            if "data" not in data or not data["data"]: return pd.DataFrame()
            
            df = pd.DataFrame(data["data"])
            # FinMind 的欄位名稱轉為標準格式
            df.rename(columns={"open": "Open", "max": "High", "min": "Low", "close": "Close", "Trading_Volume": "Volume"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            
        else:
            # 🇺🇸 走 FMP 通道 (美股)
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries=500&apikey={FMP_API_KEY}"
            resp = requests.get(url)
            if resp.status_code != 200: return pd.DataFrame()
            data = resp.json()
            if "historical" not in data: return pd.DataFrame()
            
            df = pd.DataFrame(data["historical"])
            df = df.iloc[::-1].reset_index(drop=True)
            df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        df = df.dropna(subset=["Close", "Volume"])
        if len(df) < 30: return pd.DataFrame()
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
        if is_taiwan_stock(symbol):
            # 🇹🇼 台股本益比 (FinMind)
            tw_id = clean_tw_symbol(symbol)
            start_date = (datetime.datetime.now() - datetime.timedelta(days=30)).strftime("%Y-%m-%d")
            url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPER&data_id={tw_id}&start_date={start_date}"
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and len(data["data"]) > 0:
                    return {"pe_ratio": data["data"][-1].get("PER", 15)}
            return {"pe_ratio": 15}
        else:
            # 🇺🇸 美股本益比 (FMP)
            url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
            resp = requests.get(url)
            if resp.status_code == 200 and len(resp.json()) > 0:
                pe = resp.json()[0].get("pe")
                return {"pe_ratio": pe if pe is not None else 15}
            return {"pe_ratio": 15}

    try: data = await run_in_threadpool(_fetch)
    except Exception: data = {"pe_ratio": 15}
        
    FUND_CACHE[symbol] = (data, now)
    return data

# ==========================
# 🧠 AI 綜合多因子評分系統
# ==========================
def calculate_multi_factor_score(df: pd.DataFrame, fund_data: Dict[str, float]) -> Dict[str, Any]:
    latest = df.iloc[-1]
    trend_score = 50
    if latest.get("Close") > latest.get("SMA_20", 0): trend_score += 15
    if latest.get("MACD_12_26_9", 0) > latest.get("MACDs_12_26_9", 0): trend_score += 15
    if latest.get("ADX_14", 0) > 25: trend_score += 10 
    
    mom_score = 50
    rsi = latest.get("RSI_14", 50)
    if 40 < rsi < 70: mom_score += 20
    elif rsi >= 70: mom_score -= 10 
    elif rsi <= 30: mom_score += 10 
    cci = latest.get("CCI_20_0.015", 0)
    if cci > 100: mom_score += 10
    elif cci < -100: mom_score -= 10
    
    vol_score = 50
    if latest.get("MFI_14", 50) > 50: vol_score += 25
    if latest.get("CMF_20", 0) > 0: vol_score += 25
    
    fund_score = max(0, min(100, 100 - (min(fund_data["pe_ratio"], 50) * 2)))
    
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
            "news": 65
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
        raise HTTPException(status_code=404, detail="找不到該股票資料或 API 額度用盡")

    fund_data = await fetch_fundamental(symbol)
    scoring = calculate_multi_factor_score(df, fund_data)
    
    recent_rets = df["ret"].tail(60).dropna()
    var_95 = recent_rets.quantile(0.05)
    cvar_95 = recent_rets[recent_rets <= var_95].mean() * 100 if not recent_rets.empty else -5

    last_price = float(df["Close"].iloc[-1])
    
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
