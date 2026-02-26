import os
import time
import datetime
import asyncio
import urllib.parse
import numpy as np
import pandas as pd
import pandas_ta as ta
import aiosqlite
import feedparser
import requests

from typing import Dict, Any, Tuple, List, Optional
from email.utils import parsedate_to_datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

APP_NAME = "stock-backend-quant-pro"
DATABASE_PATH = os.getenv("DATABASE_PATH", "stock_app.db")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
FMP_API_KEY = os.getenv("FMP_API_KEY", "demo")

PRICE_CACHE: Dict[str, Tuple[pd.DataFrame, float]] = {}
FUND_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}
NEWS_CACHE: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}

PRICE_CACHE_TTL = 600      
FUND_CACHE_TTL = 86400     

# ==========================
# ⚙️ FastAPI 初始化 & DB (✅ 新增投資組合資料表)
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
        # 🆕 投資組合資料表
        await db.execute("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                symbol TEXT NOT NULL,
                shares REAL NOT NULL,
                avg_cost REAL NOT NULL,
                updated_at TEXT NOT NULL,
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

# --- 請求模型 ---
class AnalysisRequest(BaseModel):
    symbol: str
    principal: float = 100000
    duration: str = "mid"

class PortfolioItem(BaseModel):
    username: str
    symbol: str
    shares: float
    avg_cost: float

# ==========================
# 🔀 核心邏輯與抓取模組
# ==========================
def is_taiwan_stock(symbol: str) -> bool:
    return symbol.replace(".TW", "").replace(".TWO", "").isdigit()

def clean_tw_symbol(symbol: str) -> str:
    return symbol.replace(".TW", "").replace(".TWO", "")

def enrich_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["ret"] = df["Close"].pct_change()
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=60, append=True) # 用於回測
    df.ta.vwma(length=20, append=True)
    df.ta.macd(append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.cci(length=20, append=True)
    df.ta.mfi(length=14, append=True) 
    df.ta.cmf(append=True) 
    df["volatility_20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    return df.dropna(subset=["SMA_20", "volatility_20", "MACD_12_26_9", "EMA_60"])

async def fetch_price_history(symbol: str) -> pd.DataFrame:
    now = time.time()
    if symbol in PRICE_CACHE:
        df, ts = PRICE_CACHE[symbol]
        if now - ts < PRICE_CACHE_TTL: return df

    def _download():
        if is_taiwan_stock(symbol):
            tw_id = clean_tw_symbol(symbol)
            start_date = (datetime.datetime.now() - datetime.timedelta(days=1000)).strftime("%Y-%m-%d")
            url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id={tw_id}&start_date={start_date}"
            resp = requests.get(url)
            if resp.status_code != 200: return pd.DataFrame()
            data = resp.json().get("data", [])
            if not data: return pd.DataFrame()
            df = pd.DataFrame(data)
            df.rename(columns={"open": "Open", "max": "High", "min": "Low", "close": "Close", "Trading_Volume": "Volume"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
        else:
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries=1000&apikey={FMP_API_KEY}"
            resp = requests.get(url)
            if resp.status_code != 200: return pd.DataFrame()
            data = resp.json().get("historical", [])
            if not data: return pd.DataFrame()
            df = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)
            df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        df = df.dropna(subset=["Close", "Volume"])
        if len(df) < 60: return pd.DataFrame()
        return enrich_indicators(df)

    df = await run_in_threadpool(_download)
    if not df.empty: PRICE_CACHE[symbol] = (df, now)
    return df

# ✅ 新增：抓取大盤基準 (Benchmark)
async def fetch_benchmark(is_tw: bool) -> pd.DataFrame:
    bench_symbol = "0050" if is_tw else "SPY"
    return await fetch_price_history(bench_symbol)

# ==========================
# 📈 回測與績效計算引擎 (✅ 新增回測功能)
# ==========================
def calculate_drawdown(returns: pd.Series) -> float:
    cum_rets = (1 + returns).cumprod()
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    return abs(drawdown.min()) * 100

def run_backtest(df: pd.DataFrame) -> Dict[str, Any]:
    # 簡單量化策略：MACD 柱狀圖 > 0 且 價格 > 60日均線 時持有
    df["signal"] = np.where((df["MACDh_12_26_9"] > 0) & (df["Close"] > df["EMA_60"]), 1, 0)
    df["strategy_ret"] = df["signal"].shift(1) * df["ret"]
    
    valid_rets = df["strategy_ret"].dropna()
    cum_ret = (1 + valid_rets).cumprod().iloc[-1] - 1 if not valid_rets.empty else 0
    bh_ret = (1 + df["ret"].dropna()).cumprod().iloc[-1] - 1 # Buy and Hold
    
    mdd = calculate_drawdown(valid_rets)
    sharpe = (valid_rets.mean() / valid_rets.std()) * np.sqrt(252) if valid_rets.std() > 0 else 0
    
    win_days = len(valid_rets[valid_rets > 0])
    total_trades = len(valid_rets[valid_rets != 0])
    win_rate = (win_days / total_trades * 100) if total_trades > 0 else 0

    return {
        "cumulative_return_pct": round(cum_ret * 100, 2),
        "buy_and_hold_return_pct": round(bh_ret * 100, 2),
        "max_drawdown_pct": round(mdd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "win_rate_pct": round(win_rate, 2)
    }

# ==========================
# 🚀 API 路由區
# ==========================

# 1️⃣ 分析與大盤比較 API
@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    symbol = request.symbol.strip().upper()
    df = await fetch_price_history(symbol)
    if df.empty: raise HTTPException(status_code=404, detail="找不到股票資料")

    # ✅ 大盤比較 (Beta & Alpha 計算)
    is_tw = is_taiwan_stock(symbol)
    bench_df = await fetch_benchmark(is_tw)
    
    beta = 1.0
    alpha = 0.0
    if not bench_df.empty:
        # 對齊日期計算共變異數
        aligned = pd.concat([df["ret"], bench_df["ret"]], axis=1).dropna()
        aligned.columns = ["stock", "bench"]
        cov = aligned.cov().iloc[0, 1]
        var = aligned["bench"].var()
        if var > 0:
            beta = cov / var
            alpha = (aligned["stock"].mean() - beta * aligned["bench"].mean()) * 252 * 100

    last_price = float(df["Close"].iloc[-1])
    recent_rets = df["ret"].tail(120).dropna()
    cvar_95 = recent_rets[recent_rets <= recent_rets.quantile(0.05)].mean()
    
    return {
        "symbol": symbol,
        "market_benchmark": "0050(台灣50)" if is_tw else "SPY(標普500)",
        "quant_metrics": {
            "beta": round(beta, 2),          # 衡量與大盤的連動性 (>1代表比大盤活潑)
            "annual_alpha_pct": round(alpha, 2), # 衡量超越大盤的絕對報酬
            "current_price": last_price,
            "stop_loss_suggested": round(last_price * (1 + (cvar_95 if pd.notna(cvar_95) else -0.05)), 2)
        }
    }

# 2️⃣ 回測 API (✅ 新增)
@app.get("/api/backtest/{symbol}")
async def backtest_endpoint(symbol: str):
    df = await fetch_price_history(symbol.upper())
    if df.empty: raise HTTPException(status_code=404, detail="找不到股票資料")
    
    bt_result = run_backtest(df)
    return {"symbol": symbol.upper(), "backtest_3yr": bt_result}

# 3️⃣ 投資組合管理 API (✅ 新增)
@app.post("/api/portfolio")
async def add_portfolio(item: PortfolioItem):
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("""
            INSERT INTO portfolios (username, symbol, shares, avg_cost, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(username, symbol) DO UPDATE SET
            shares = excluded.shares, avg_cost = excluded.avg_cost, updated_at = excluded.updated_at
        """, (item.username, item.symbol.upper(), item.shares, item.avg_cost, datetime.datetime.now().isoformat()))
        await db.commit()
    return {"message": "投資組合已更新"}

@app.get("/api/portfolio/{username}")
async def get_portfolio(username: str):
    async with aiosqlite.connect(DATABASE_PATH) as db:
        async with db.execute("SELECT symbol, shares, avg_cost FROM portfolios WHERE username = ?", (username,)) as cursor:
            rows = await cursor.fetchall()
            
    if not rows: return {"username": username, "total_value": 0, "positions": []}
    
    positions = []
    total_value = 0
    total_cost = 0
    
    for row in rows:
        symbol, shares, avg_cost = row
        df = await fetch_price_history(symbol)
        current_price = float(df["Close"].iloc[-1]) if not df.empty else avg_cost
        
        market_value = current_price * shares
        cost_value = avg_cost * shares
        unrealized_pl = market_value - cost_value
        unrealized_pl_pct = (unrealized_pl / cost_value * 100) if cost_value > 0 else 0
        
        total_value += market_value
        total_cost += cost_value
        
        positions.append({
            "symbol": symbol,
            "shares": shares,
            "avg_cost": avg_cost,
            "current_price": current_price,
            "market_value": round(market_value, 2),
            "unrealized_pl": round(unrealized_pl, 2),
            "unrealized_pl_pct": round(unrealized_pl_pct, 2)
        })
        
    total_pl_pct = ((total_value - total_cost) / total_cost * 100) if total_cost > 0 else 0
    
    return {
        "username": username,
        "summary": {
            "total_market_value": round(total_value, 2),
            "total_cost": round(total_cost, 2),
            "total_unrealized_pl": round(total_value - total_cost, 2),
            "total_return_pct": round(total_pl_pct, 2)
        },
        "positions": positions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
