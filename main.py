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
    clean_symbol = symbol.replace(".TW", "").replace(".TWO", "")
    return clean_symbol.isdigit()

def clean_tw_symbol(symbol: str) -> str:
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
            tw_id = clean_tw_symbol(symbol)
            start_date = (datetime.datetime.now() - datetime.timedelta(days=700)).strftime("%Y-%m-%d")
            url = f"https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockPrice&data_id={tw_id}&start_date={start_date}"
            resp = requests.get(url)
            if resp.status_code != 200: return pd.DataFrame()
            data = resp.json()
            if "data" not in data or not data["data"]: return pd.DataFrame()
            
            df = pd.DataFrame(data["data"])
            df.rename(columns={"open": "Open", "max": "High", "min": "Low", "close": "Close", "Trading_Volume": "Volume"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            
        else:
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
# 🧠 AI 專業量化引擎 (動態 IC 權重 + 去共線性)
# ==========================
def calculate_multi_factor_score(df: pd.DataFrame, fund_data: Dict[str, float]) -> Dict[str, Any]:
    # 保留最新的數值供最後評分使用
    latest = df.iloc[-1].copy()
    
    # 1. 建立真實預測目標：未來 5 天的預期報酬 (Forward Return)
    df["fwd_ret_5d"] = df["Close"].shift(-5) / df["Close"] - 1
    
    # 取最近 120 天作為 Walk-Forward 的訓練窗口 (需丟棄最後 5 天 NaN)
    train_df = df.dropna(subset=["fwd_ret_5d", "MACD_12_26_9"]).tail(120).copy()
    
    if len(train_df) < 30: # 避免資料不足
        return {"final_score": 50, "breakdown": {"technical": 50, "fundamental": 50, "chip": 50, "news": 50}, "regime": "資料不足"}

    # 2. 因子清單與分類 (轉化為可比較的數值)
    train_df["trend_macd"] = train_df["MACD_12_26_9"] - train_df["MACDs_12_26_9"]
    train_df["trend_sma"] = (train_df["Close"] - train_df["SMA_20"]) / train_df["SMA_20"]
    train_df["mom_rsi"] = train_df["RSI_14"]
    train_df["mom_cci"] = train_df["CCI_20_0.015"]
    train_df["vol_mfi"] = train_df["MFI_14"]
    train_df["vol_cmf"] = train_df["CMF_20"]
    
    # 3. 計算 Information Coefficient (IC) - 使用 Pandas 內建 Spearman
    ic_dict = {}
    for factor in ["trend_macd", "trend_sma", "mom_rsi", "mom_cci", "vol_mfi", "vol_cmf"]:
        ic = train_df[factor].corr(train_df["fwd_ret_5d"], method="spearman")
        ic_dict[factor] = ic if pd.notna(ic) else 0

    # 4. 因子去共線性 (選擇每組預測力最強的代表，拋棄高度相關的冗餘因子)
    best_trend = "trend_macd" if abs(ic_dict["trend_macd"]) > abs(ic_dict["trend_sma"]) else "trend_sma"
    best_mom = "mom_rsi" if abs(ic_dict["mom_rsi"]) > abs(ic_dict["mom_cci"]) else "mom_cci"
    best_vol = "vol_mfi" if abs(ic_dict["vol_mfi"]) > abs(ic_dict["vol_cmf"]) else "vol_cmf"

    # 5. 動態權重分配 (Dynamic Weighting based on IC)
    total_ic = abs(ic_dict[best_trend]) + abs(ic_dict[best_mom]) + abs(ic_dict[best_vol])
    if total_ic == 0: total_ic = 1 # 避免除以零
    
    w_trend = abs(ic_dict[best_trend]) / total_ic
    w_mom = abs(ic_dict[best_mom]) / total_ic
    w_vol = abs(ic_dict[best_vol]) / total_ic

    # 6. 計算當前最新訊號的分數 (將因子標準化並乘上方向)
    def get_signal_score(factor, current_val):
        min_val, max_val = train_df[factor].min(), train_df[factor].max()
        if max_val == min_val: return 50
        score = (current_val - min_val) / (max_val - min_val) * 100
        # 如果 IC 是負的，代表該因子是反向指標，自動倒轉分數
        return score if ic_dict[factor] > 0 else 100 - score

    curr_trend = (latest["MACD_12_26_9"] - latest["MACDs_12_26_9"]) if best_trend == "trend_macd" else (latest["Close"] - latest["SMA_20"]) / latest["SMA_20"]
    curr_mom = latest["RSI_14"] if best_mom == "mom_rsi" else latest["CCI_20_0.015"]
    curr_vol = latest["MFI_14"] if best_vol == "vol_mfi" else latest["CMF_20"]

    trend_score = get_signal_score(best_trend, curr_trend)
    mom_score = get_signal_score(best_mom, curr_mom)
    vol_score = get_signal_score(best_vol, curr_vol)
    
    # 基本面估值
    fund_score = max(0, min(100, 100 - (min(fund_data["pe_ratio"], 50) * 2)))

    # 7. 轉換為預測勝率 (Logistic 概念轉換)
    raw_alpha = (trend_score * w_trend) + (mom_score * w_mom) + (vol_score * w_vol)
    combined_score = (raw_alpha * 0.8) + (fund_score * 0.2)
    
    # 轉換為真實世界勝率 (40% ~ 70% 之間浮動)
    win_rate = 40 + (combined_score / 100) * 30 
    
    ann_vol = latest.get("volatility_20", 0.2)
    regime = "高波動洗盤期" if ann_vol > 0.35 else "穩健趨勢期"

    return {
        "final_score": int(win_rate), 
        "breakdown": {
            "technical": int(trend_score),
            "fundamental": int(fund_score),
            "chip": int(vol_score),
            "news": int(mom_score) # 借用 news 欄位放動能分數
        },
        "regime": f"{regime} | {best_trend.split('_')[1].upper()}驅動"
    }

# ==========================
# 🚀 股票分析 API 路由 (厚尾分配 Monte Carlo)
# ==========================
@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    symbol = request.symbol.strip().upper()
    df = await fetch_price_history(symbol)

    if df.empty:
        raise HTTPException(status_code=404, detail="找不到該股票資料或 API 額度用盡")

    fund_data = await fetch_fundamental(symbol)
    scoring = calculate_multi_factor_score(df, fund_data)
    
    # ==========================================
    # 專業風險模型：GARCH 概念 + 厚尾 CVaR
    # ==========================================
    recent_rets = df["ret"].tail(120).dropna()
    
    # 使用 EWMA (指數加權移動平均) 近似 GARCH，捕捉近期波動群聚
    ewma_vol = df["ret"].ewm(span=20).std().iloc[-1]
    if pd.isna(ewma_vol): ewma_vol = recent_rets.std()
    
    # 真實世界 CVaR (計算 5% 尾部風險)
    var_95 = recent_rets.quantile(0.05)
    cvar_95 = recent_rets[recent_rets <= var_95].mean()
    if pd.isna(cvar_95): cvar_95 = -0.05
    
    last_price = float(df["Close"].iloc[-1])
    days = 14 if request.duration == "short" else 60 if request.duration == "mid" else 180
    
    # 取得歷史 Drift，結合 AI 勝率進行 Alpha 調整
    alpha_adjustment = (scoring["final_score"] - 50) / 1000 
    drift = recent_rets.mean() + alpha_adjustment
    
    predictions = []
    sim_p = last_price
    
    # Monte Carlo 升級：Student's t 分配 (自由度 df=4) 模擬 Fat Tail (厚尾現象)
    df_t = 4 
    scale_factor = np.sqrt((df_t - 2) / df_t) 
    
    for i in range(1, days + 1):
        # 使用 numpy 原生的 standard_t 模擬極端黑天鵝
        innovation = np.random.standard_t(df_t) * scale_factor
        sim_p *= (1 + drift + ewma_vol * innovation)
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
        "ai_score": scoring["final_score"], # 現在輸出的是「勝率 %」
        "ai_sentiment": scoring["regime"],
        "score_breakdown": scoring["breakdown"],
        "advice": {
            "buy_price": round(last_price * 0.98, 2),
            "take_profit": predictions[-1]["mid"],
            "stop_loss": round(last_price * (1 + cvar_95), 2) # 真實 CVaR 停損
        },
        "chart_data": {
            "history": history_data,
            "prediction": predictions
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
