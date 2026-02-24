import os
import re
import time
import sqlite3
import datetime
import asyncio
import urllib.parse  # 用於修復 URL 編碼問題
from typing import Optional, List, Dict, Any
from email.utils import parsedate_to_datetime
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, HTTPException, status, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ====== 可選：真實新聞 RSS 解析 ======
try:
    import feedparser  # type: ignore
except Exception:
    feedparser = None


# ==========================
# 0) 設定與快取
# ==========================
APP_NAME = "stock-backend"
DATABASE_PATH = os.getenv("DATABASE_PATH", "stock_app.db")
VERCEL_FRONTEND = os.getenv("VERCEL_FRONTEND", "https://stock-frontend-theta.vercel.app")

# 全域新聞快取，用於每小時自動更新
NEWS_CACHE = {
    "data": [],
    "last_updated": None
}

# 背景定時任務：每 3600 秒更新一次新聞
async def update_news_periodically():
    while True:
        try:
            print(f"[{datetime.datetime.now()}] 執行每小時新聞自動更新...")
            # 抓取預設財經新聞
            latest_news = fetch_real_news("全球市場 財經", limit=15)
            if latest_news:
                NEWS_CACHE["data"] = latest_news
                NEWS_CACHE["last_updated"] = datetime.datetime.now().isoformat()
        except Exception as e:
            print(f"自動更新任務出錯: {e}")
        
        await asyncio.sleep(3600)  # 等待一小時

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 伺服器啟動時執行
    task = asyncio.create_task(update_news_periodically())
    yield
    # 伺服器關閉時執行
    task.cancel()

app = FastAPI(title=APP_NAME, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================
# 1) DB 初始化
# ==========================
def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
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
    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            symbol TEXT NOT NULL,
            shares INTEGER NOT NULL,
            avg_cost REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()


# ==========================
# 2) Pydantic Models
# ==========================
class AnalysisRequest(BaseModel):
    symbol: str
    principal: float
    strategy: str
    duration: str  # day/short/mid/long


class FavoriteReq(BaseModel):
    symbol: str


class PortfolioItem(BaseModel):
    symbol: str
    shares: int
    cost: float


# ==========================
# 3) 指標計算邏輯 (保持原樣)
# ==========================
def fetch_price_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
        if df is None or df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close"])
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.bfill().fillna(50)

def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist

def compute_bollinger(close: pd.Series, window: int = 20, k: float = 2.0):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + k * std
    lower = ma - k * std
    return ma, upper, lower

def compute_kd(df: pd.DataFrame, k_period: int = 9, d_period: int = 3):
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / d_period, adjust=False).mean()
    d = k.ewm(alpha=1 / d_period, adjust=False).mean()
    return k.fillna(50), d.fillna(50)

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ==========================
# 4) 四大面向評分 (保持原樣)
# ==========================
def score_technical(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["Close"]
    rsi = compute_rsi(close)
    macd, sig, hist = compute_macd(close)
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    bb_ma, bb_up, bb_low = compute_bollinger(close)
    k, d = compute_kd(df)

    rsi_v = float(rsi.iloc[-1])
    macd_v = float(macd.iloc[-1])
    sig_v = float(sig.iloc[-1])
    hist_v = float(hist.iloc[-1])
    ma20_v = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else float(close.iloc[-1])
    ma60_v = float(ma60.iloc[-1]) if not np.isnan(ma60.iloc[-1]) else float(close.iloc[-1])
    ma120_v = float(ma120.iloc[-1]) if not np.isnan(ma120.iloc[-1]) else float(close.iloc[-1])
    k_v = float(k.iloc[-1]); d_v = float(d.iloc[-1])
    price = float(close.iloc[-1])
    
    s = 50.0
    if price > ma20_v: s += 8
    if ma20_v > ma60_v: s += 10
    if ma60_v > ma120_v: s += 10
    if macd_v > sig_v: s += 8
    if hist_v > 0: s += 4
    if rsi_v < 30: s += 6
    elif rsi_v > 70: s -= 6
    if k_v > d_v: s += 4
    if k_v < 20: s += 2
    if k_v > 80: s -= 2
    
    ret = close.pct_change().dropna()
    vol = float(ret.tail(60).std() * np.sqrt(252)) if len(ret) >= 30 else 0.35
    if vol < 0.25: s += 6
    elif vol > 0.6: s -= 6

    return {
        "score": int(round(clamp(s, 0, 100))),
        "rsi": round(rsi_v, 2), "macd": round(macd_v, 4), "macd_signal": round(sig_v, 4),
        "ma20": round(ma20_v, 2), "ma60": round(ma60_v, 2), "ma120": round(ma120_v, 2),
        "bb_mid": round(float(bb_ma.iloc[-1]) if not np.isnan(bb_ma.iloc[-1]) else price, 2),
        "bb_upper": round(float(bb_up.iloc[-1]) if not np.isnan(bb_up.iloc[-1]) else price, 2),
        "bb_lower": round(float(bb_low.iloc[-1]) if not np.isnan(bb_low.iloc[-1]) else price, 2),
        "kd_k": round(k_v, 2), "kd_d": round(d_v, 2), "annual_volatility": round(vol, 4),
    }

def score_fundamental(symbol: str) -> Dict[str, Any]:
    s = 50.0
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
    except: info = {}

    roe = info.get("returnOnEquity")
    pm = info.get("profitMargins")
    rev_g = info.get("revenueGrowth"); earn_g = info.get("earningsGrowth")
    debt = info.get("debtToEquity"); pe = info.get("forwardPE"); pb = info.get("priceToBook")

    if isinstance(roe, (int, float)):
        if roe >= 0.2: s += 12
        elif roe < 0.05: s -= 6
    if isinstance(pm, (int, float)) and pm >= 0.15: s += 8
    if isinstance(debt, (int, float)) and debt < 50: s += 4
    
    return {
        "score": int(round(clamp(s, 0, 100))),
        "roe": roe, "profit_margin": pm, "revenue_growth": rev_g,
        "earnings_growth": earn_g, "debt_to_equity": debt,
        "forward_pe": pe, "price_to_book": pb,
    }

def score_chip_proxy(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["Close"]; vol = df["Volume"].fillna(0)
    obv = (np.sign(close.diff().fillna(0)) * vol).cumsum()
    y = obv.tail(20).values; x = np.arange(len(y))
    den = ((x - x.mean()) ** 2).sum() or 1e-9
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / den)
    v5 = float(vol.tail(5).mean()); v20 = float(vol.tail(20).mean())
    vol_ratio = (v5 / v20) if v20 > 0 else 1.0
    s = 50.0 + (12 if slope > 0 else -6) + (10 if vol_ratio >= 1.3 else -6)
    return {"score": int(round(clamp(s, 0, 100))), "obv_slope_20d": round(slope, 4), "volume_ratio_5v20": round(vol_ratio, 4)}

def score_news_sentiment(news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    s = 50.0
    pos_kw = ["上調", "成長", "創新高", "強勁", "利多", "看好", "獲利", "大漲", "買超", "調升"]
    neg_kw = ["下調", "衰退", "利空", "大跌", "賣超", "風險", "崩跌", "裁員", "警告", "疲弱"]
    titles = " ".join([(x.get("title") or "") for x in news_items]).lower()
    pos = sum([titles.count(k.lower()) for k in pos_kw])
    neg = sum([titles.count(k.lower()) for k in neg_kw])
    s = s + (pos * 2.5) - (neg * 3.0)
    return {"score": int(round(clamp(s, 0, 100))), "pos_hits": pos, "neg_hits": neg}

def composite_score(tech: int, fund: int, chip: int, news: int) -> int:
    return int(round(clamp(0.35 * tech + 0.25 * fund + 0.20 * chip + 0.20 * news, 0, 100)))

def sentiment_text(score: int) -> str:
    if score >= 80: return "強力看多"
    if score >= 60: return "偏多"
    if score >= 40: return "中立"
    return "偏空"

# ==========================
# 5) 估算與預測邏輯 (保持原樣)
# ==========================
def estimate_roi(cost: float, df: pd.DataFrame) -> Dict[str, Any]:
    ret = df["Close"].pct_change().dropna()
    if ret.empty: return {k: {"pct": 0.0, "amt": 0} for k in ["day", "short", "mid", "long"]}
    mu = float(ret.tail(120).mean()); sigma = float(ret.tail(120).std())
    horizons = {"day": 1, "short": 5, "mid": 60, "long": 252}
    out = {}
    for k, d in horizons.items():
        pct = clamp(mu * d, -max(0.08, 2.5 * sigma * np.sqrt(d)), max(0.08, 2.5 * sigma * np.sqrt(d)))
        out[k] = {"pct": round(pct * 100, 2), "amt": int(round(cost * pct))}
    return out

def extreme_risk_95(cost: float, df: pd.DataFrame, horizon_days: int = 60) -> Dict[str, Any]:
    ret = df["Close"].pct_change().dropna()
    if ret.empty: return {"max_loss_amt": 0, "max_loss_pct": 0.0, "pessimistic_price": 0}
    var_h = float(ret.quantile(0.05)) * np.sqrt(horizon_days)
    price = float(df["Close"].iloc[-1])
    return {
        "max_loss_amt": int(round(cost * abs(var_h))),
        "max_loss_pct": round(abs(var_h) * 100, 2),
        "pessimistic_price": round(price * (1 + var_h), 2),
        "var_1d_pct": round(abs(float(ret.quantile(0.05))) * 100, 2),
    }

def band_trade_prices(current_price: float) -> Dict[str, float]:
    return {"buy_price": round(current_price, 2), "take_profit": round(current_price * 1.20, 2), "stop_loss": round(current_price * 0.90, 2)}

def build_chart_data(df: pd.DataFrame, future_days: int = 30) -> Dict[str, Any]:
    df_use = df.tail(240).copy(); close = df_use["Close"].astype(float)
    y = np.log(close.values); x = np.arange(len(y))
    den = ((x - x.mean()) ** 2).sum() or 1e-9
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / den)
    intercept = float(y.mean() - slope * x.mean())
    resid_std = float(np.std(y - (intercept + slope * x))) or 0.02
    history = [{"date": d.strftime("%Y-%m-%d"), "price": round(float(p), 2)} for d, p in zip(df_use.index, close.values)]
    pred = []
    for i in range(1, future_days + 1):
        yi = intercept + slope * (len(y) - 1 + i)
        pred.append({
            "date": (df_use.index[-1] + datetime.timedelta(days=i)).strftime("%Y-%m-%d"),
            "mid": round(float(np.exp(yi)), 2),
            "upper": round(float(np.exp(yi + 1.96 * resid_std)), 2),
            "lower": round(float(np.exp(yi - 1.96 * resid_std)), 2),
        })
    return {"history": history, "prediction": pred}


# ==========================
# 6) 新聞抓取邏輯
# ==========================
def parse_dt(entry: dict) -> Optional[datetime.datetime]:
    if entry.get("published_parsed"):
        try: return datetime.datetime(*entry["published_parsed"][:6])
        except: pass
    if entry.get("published"):
        try: return parsedate_to_datetime(entry["published"])
        except: return None
    return None

def fetch_real_news(query: str, limit: int = 12) -> List[Dict[str, Any]]:
    if feedparser is None: return []
    safe_query = urllib.parse.quote(query.strip())
    rss_url = f"https://news.google.com/rss/search?q={safe_query}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        print(f"RSS Fetch Error: {e}"); return []

    items = []
    for e in feed.entries[:limit * 2]:
        dt = parse_dt(e)
        title = getattr(e, "title", "")
        tag = "產業"
        if any(k in title.lower() for k in ["風險", "戰", "通膨", "升息"]): tag = "風險"
        if any(k in title.lower() for k in ["法說", "財報", "營收"]): tag = "評論"
        items.append({
            "tag": tag, "time": getattr(e, "published", "") or (dt.strftime("%Y-%m-%d %H:%M") if dt else ""),
            "published_at": dt.isoformat() if dt else "", "title": title, "url": getattr(e, "link", ""), "source": "Google News",
        })
    seen = set(); uniq = []
    for it in items:
        if it["url"] not in seen:
            seen.add(it["url"]); uniq.append(it)
    uniq.sort(key=lambda x: x["published_at"], reverse=True)
    return uniq[:limit]


# ==========================
# 7) 路由 API
# ==========================
@app.get("/health")
async def health():
    return {"status": "ok", "server_time_utc": datetime.datetime.utcnow().isoformat(), "cache_updated": NEWS_CACHE["last_updated"]}

@app.get("/api/news")
async def api_news(q: str = Query(None), limit: int = Query(10, ge=1, le=20)):
    # 如果沒給關鍵字或是預設關鍵字，直接回傳快取
    if q is None or q == "全球市場 財經":
        if NEWS_CACHE["data"]:
            return NEWS_CACHE["data"][:limit]
        return fetch_real_news("全球市場 財經", limit=limit)
    # 搜尋特定股票則即時抓取
    return fetch_real_news(q, limit=limit)

@app.post("/api/analyze")
async def analyze_stock(request: AnalysisRequest):
    symbol = request.symbol.strip().upper()
    df = fetch_price_history(symbol)
    if df.empty: raise HTTPException(status_code=404, detail="找不到股票資料")
    if len(df) < 60: raise HTTPException(status_code=400, detail="資料筆數不足")

    current_price = float(df["Close"].iloc[-1])
    tech = score_technical(df)
    fund = score_fundamental(symbol)
    chip = score_chip_proxy(df)
    news_items = fetch_real_news(symbol, limit=10)
    news_score = score_news_sentiment(news_items)
    overall = composite_score(tech["score"], fund["score"], chip["score"], news_score["score"])
    
    principal = float(request.principal)
    max_shares = int(principal // current_price)
    total_cost = float(max_shares * current_price)

    return {
        "symbol": symbol, "price": round(current_price, 2), "ai_score": overall, "ai_sentiment": sentiment_text(overall),
        "score_breakdown": {"technical": tech, "fundamental": fund, "chip": chip, "news": news_score},
        "money_management": {"principal": principal, "max_shares": max_shares, "total_cost": total_cost},
        "advice": band_trade_prices(current_price), "roi_estimates": estimate_roi(total_cost, df),
        "risk_analysis": extreme_risk_95(total_cost, df), "chart_data": build_chart_data(df)
    }

@app.get("/api/favorites")
async def get_favorites():
    user = "guest"
    conn = get_db(); c = conn.cursor()
    c.execute("SELECT symbol, created_at FROM favorites WHERE username=? ORDER BY created_at DESC", (user,))
    rows = c.fetchall(); conn.close()
    return [{"symbol": r["symbol"], "created_at": r["created_at"]} for r in rows]

@app.post("/api/favorites")
async def add_favorite(req: FavoriteReq):
    user = "guest"; sym = (req.symbol or "").strip().upper()
    conn = get_db(); c = conn.cursor()
    c.execute("INSERT OR IGNORE INTO favorites (username, symbol, created_at) VALUES (?, ?, ?)",
              (user, sym, datetime.datetime.utcnow().isoformat()))
    conn.commit(); conn.close()
    return {"message": "ok", "symbol": sym}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
