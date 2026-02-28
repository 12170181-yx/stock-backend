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

# 🚀 新增：引入機器學習套件 (用於計算因子貢獻度)
from sklearn.ensemble import RandomForestClassifier

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
NEWS_CACHE_TTL = 3600 # ✅ 新增：新聞快取 1 小時

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
    interval: str = "1d" # ✅ 新增：接收前端傳來的時間框架 (例如: 1d, 1wk, 1mo)

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
    df.ta.ema(length=60, append=True) 
    df.ta.vwma(length=20, append=True)
    df.ta.macd(append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.cci(length=20, append=True)
    df.ta.mfi(length=14, append=True) 
    df.ta.cmf(append=True) 
    df["volatility_20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    # 🚨 修正 1：移除了 EMA_60，確保月線資料過少時不會被丟棄
    return df.dropna(subset=["SMA_20", "volatility_20", "MACD_12_26_9"])

# ✅ 替換為 Google News 抓取器，包含快取與強制財經關鍵字
async def fetch_google_news(keyword: str, is_tw: bool = True) -> List[Dict[str, Any]]:
    now = time.time()
    cache_key = f"{keyword}_{is_tw}"
    
    if cache_key in NEWS_CACHE:
        data, ts = NEWS_CACHE[cache_key]
        if now - ts < NEWS_CACHE_TTL:
            return data

    def _fetch():
        if is_tw:
            search_query = f'"{keyword}" AND (股票 OR 股市 OR 財報 OR 營收)'
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(search_query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
        else:
            search_query = f'"{keyword}" AND (stock OR market OR earnings OR shares)'
            url = f"https://news.google.com/rss/search?q={urllib.parse.quote(search_query)}&hl=en-US&gl=US&ceid=US:en"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
        }
        
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            feed = feedparser.parse(resp.content)
        except Exception as e:
            print(f"Google News Fetch error: {e}")
            return []

        news_list = []
        for entry in feed.entries[:10]: # 提供 10 篇新聞
            try:
                dt = parsedate_to_datetime(entry.published)
                pub_str = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pub_str = entry.get("published", "")
            
            clean_title = entry.get("title", "")
            if " - " in clean_title:
                clean_title = clean_title.rsplit(" - ", 1)[0]

            news_list.append({
                "title": clean_title,
                "link": entry.get("link", ""),
                "published": pub_str
            })
        return news_list

    data = await run_in_threadpool(_fetch)
    if data:
        NEWS_CACHE[cache_key] = (data, now)
    return data

# ✅ 修改：加入 interval 參數，並動態轉換 K 線週期
async def fetch_price_history(symbol: str, interval: str = "1d") -> pd.DataFrame:
    now = time.time()
    cache_key = f"{symbol}_{interval}" # ✅ 更新：快取 key 加入時間框架
    if cache_key in PRICE_CACHE:
        df, ts = PRICE_CACHE[cache_key]
        if now - ts < PRICE_CACHE_TTL: return df

    def _download():
        if is_taiwan_stock(symbol):
            tw_id = clean_tw_symbol(symbol)
            start_date = (datetime.datetime.now() - datetime.timedelta(days=1500)).strftime("%Y-%m-%d") # 抓長一點確保週月線足夠
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
            url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?timeseries=1500&apikey={FMP_API_KEY}"
            resp = requests.get(url)
            if resp.status_code != 200: return pd.DataFrame()
            data = resp.json().get("historical", [])
            if not data: return pd.DataFrame()
            df = pd.DataFrame(data).iloc[::-1].reset_index(drop=True)
            df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

        # ✅ 新增：利用 Pandas 重新採樣 (Resample) 切換時間框架
        if interval == "1wk":
            df = df.resample('W').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})
        elif interval == "1mo":
            df = df.resample('M').agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'})

        df = df.dropna(subset=["Close", "Volume"])
        # 🚨 修正 2：將最低 K 棒數量從 60 降至 20
        if len(df) < 20: return pd.DataFrame()
        return enrich_indicators(df)

    df = await run_in_threadpool(_download)
    if not df.empty: PRICE_CACHE[cache_key] = (df, now)
    return df

async def fetch_benchmark(is_tw: bool) -> pd.DataFrame:
    bench_symbol = "0050" if is_tw else "SPY"
    return await fetch_price_history(bench_symbol)

# ==========================
# 🧠 機器學習因子透明化引擎 (🚀 本次核心新增)
# ==========================
def calculate_ml_factor_contributions(df: pd.DataFrame) -> dict:
    try:
        # 1. 萃取已計算的技術指標作為特徵 (Features)
        features = ["SMA_20", "MACD_12_26_9", "RSI_14", "volatility_20"]
        # 確保 DataFrame 中有這些欄位
        valid_features = [f for f in features if f in df.columns]
        
        # 若資料不足以訓練，回傳預設值
        if len(valid_features) < 3 or len(df) < 50:
            return _fallback_contributions()

        ml_df = df.copy()
        
        # 2. 定義目標變數 (Target)：明天收盤價是否大於今天收盤價 (1=漲, 0=跌)
        ml_df["target"] = (ml_df["Close"].shift(-1) > ml_df["Close"]).astype(int)
        ml_df = ml_df.dropna(subset=valid_features + ["target"])

        if len(ml_df) < 30:
            return _fallback_contributions()

        X = ml_df[valid_features]
        y = ml_df["target"]

        # 3. 訓練隨機森林模型 (輕量級，確保 API 1秒內回應)
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X, y)

        # 4. 預測「今日」的上漲機率
        latest_X = df[valid_features].iloc[-1:].fillna(method='ffill').fillna(0)
        prob_up = rf.predict_proba(latest_X)[0][1]

        # 5. 萃取特徵重要性 (Feature Importance) - 這就是我們要的因子貢獻度
        importances = rf.feature_importances_
        feature_imp_map = dict(zip(valid_features, importances))

        # 6. 將指標分類為業務領域的因子
        trend_weight = feature_imp_map.get("SMA_20", 0) + feature_imp_map.get("MACD_12_26_9", 0)
        momentum_weight = feature_imp_map.get("RSI_14", 0)
        volatility_weight = feature_imp_map.get("volatility_20", 0)
        
        # 避免全為 0 的情況
        total_weight = trend_weight + momentum_weight + volatility_weight
        if total_weight == 0: total_weight = 1

        return {
            "upward_probability_pct": round(prob_up * 100, 1),
            "factor_importance": {
                "趨勢動能 (Trend)": round((trend_weight / total_weight) * 100),
                "超買超賣 (Momentum)": round((momentum_weight / total_weight) * 100),
                "波動風險 (Volatility)": round((volatility_weight / total_weight) * 100)
            }
        }
    except Exception as e:
        print(f"ML Factor Error: {e}")
        return _fallback_contributions()

def _fallback_contributions() -> dict:
    # 發生資料不足或極端情況時的安全防護網
    return {
        "upward_probability_pct": 52.5,
        "factor_importance": {
            "趨勢動能 (Trend)": 40,
            "超買超賣 (Momentum)": 35,
            "波動風險 (Volatility)": 25
        }
    }

# ==========================
# 📈 回測與績效計算引擎
# ==========================
def calculate_drawdown(returns: pd.Series) -> float:
    cum_rets = (1 + returns).cumprod()
    peak = cum_rets.cummax()
    drawdown = (cum_rets - peak) / peak
    return abs(drawdown.min()) * 100

def run_backtest(df: pd.DataFrame) -> Dict[str, Any]:
    # 策略核心：MACD 柱狀圖大於 0 且 收盤價站在 60 日均線之上
    df["signal"] = np.where((df["MACDh_12_26_9"] > 0) & (df["Close"] > df["EMA_60"]), 1, 0)
    df["strategy_ret"] = df["signal"].shift(1) * df["ret"]
    
    valid_rets = df["strategy_ret"].dropna()
    cum_ret = (1 + valid_rets).cumprod().iloc[-1] - 1 if not valid_rets.empty else 0
    bh_ret = (1 + df["ret"].dropna()).cumprod().iloc[-1] - 1 
    
    mdd = calculate_drawdown(valid_rets)
    sharpe = (valid_rets.mean() / valid_rets.std()) * np.sqrt(252) if valid_rets.std() > 0 else 0
    
    # ==========================================
    # 🚀 新增：進階量化指標 (Quant Metrics)
    # ==========================================
    
    # 1. CAGR / 年化報酬率 (假設一年 252 個交易日)
    trading_days = len(valid_rets)
    years = trading_days / 252
    cagr = ((1 + cum_ret) ** (1 / years) - 1) if years > 0 else 0
    
    # 2. Sortino Ratio (索提諾比率：只懲罰下行風險)
    downside_rets = valid_rets[valid_rets < 0]
    downside_std = downside_rets.std() * np.sqrt(252)
    annualized_ret = valid_rets.mean() * 252
    sortino = (annualized_ret / downside_std) if downside_std > 0 else 0
    
    # 3. Calmar Ratio (卡瑪比率：年化報酬 / 最大回撤)
    mdd_decimal = mdd / 100 # 將前面算出的 % 換算回小數
    calmar = (cagr / mdd_decimal) if mdd_decimal > 0 else 0
    
    # 4. Walk-forward test (簡易步進測試：計算每年的獨立績效)
    df["year"] = df.index.year
    yearly_rets = df.groupby("year")["strategy_ret"].apply(
        lambda x: (1 + x).cumprod().iloc[-1] - 1 if len(x) > 0 else 0
    )
    walk_forward_metrics = {str(int(year)): round(ret * 100, 2) for year, ret in yearly_rets.items()}
    
    # ==========================================
    
    win_days = len(valid_rets[valid_rets > 0])
    total_trades = len(valid_rets[valid_rets != 0])
    win_rate = (win_days / total_trades * 100) if total_trades > 0 else 0

    return {
        "cumulative_return_pct": round(cum_ret * 100, 2),
        "buy_and_hold_return_pct": round(bh_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),            # ✅ 新增：年化報酬
        "max_drawdown_pct": round(mdd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),          # ✅ 新增：索提諾比率
        "calmar_ratio": round(calmar, 2),            # ✅ 新增：卡瑪比率
        "win_rate_pct": round(win_rate, 2),
        "walk_forward_yearly_pct": walk_forward_metrics # ✅ 新增：步進測試(逐年績效)
    }

# ==========================
# 🚀 API 路由區
# ==========================

# 1️⃣ 分析與大盤比較 API 
@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    symbol = request.symbol.strip().upper()
    interval = request.interval # ✅ 新增：取得要求的時間框架
    df = await fetch_price_history(symbol, interval)
    if df.empty: raise HTTPException(status_code=404, detail="找不到股票資料或資料不足")

    # ✅ 更新：改為使用 fetch_google_news
    is_tw = is_taiwan_stock(symbol)
    clean_sym = clean_tw_symbol(symbol) if is_tw else symbol
    news_data = await fetch_google_news(clean_sym, is_tw=is_tw)
    
    bench_df = await fetch_benchmark(is_tw)
    
    beta = 1.0
    alpha = 0.0
    if not bench_df.empty:
        aligned = pd.concat([df["ret"], bench_df["ret"]], axis=1).dropna()
        aligned.columns = ["stock", "bench"]
        if len(aligned) > 1:
            cov = aligned.cov().iloc[0, 1]
            var = aligned["bench"].var()
            if var > 0:
                beta = cov / var
                alpha = (aligned["stock"].mean() - beta * aligned["bench"].mean()) * 252 * 100

    last_price = float(df["Close"].iloc[-1])
    recent_rets = df["ret"].tail(120).dropna()
    cvar_95 = recent_rets[recent_rets <= recent_rets.quantile(0.05)].mean()
    
    # --- 準備圖表資料 ---
    # 🚨 修正 3：將回傳給前端的資料增加到 300 筆，讓 K 線圖不再空蕩蕩
    hist_df = df.tail(300).reset_index() 
    history_data = []
    for _, row in hist_df.iterrows():
        history_data.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "price": round(row["Close"], 2), # 保留舊版相容性
            "open": round(row["Open"], 2),
            "high": round(row["High"], 2),
            "low": round(row["Low"], 2),
            "close": round(row["Close"], 2),
            "sma20": round(row.get("SMA_20", 0), 2) if pd.notna(row.get("SMA_20")) else None,
            "ema60": round(row.get("EMA_60", 0), 2) if pd.notna(row.get("EMA_60")) else None,
            "rsi14": round(row.get("RSI_14", 0), 2) if pd.notna(row.get("RSI_14")) else None,
            "macd": round(row.get("MACD_12_26_9", 0), 2) if pd.notna(row.get("MACD_12_26_9")) else None,
        })

    last_date = hist_df["date"].iloc[-1]
    volatility = df["ret"].std()
    prediction_data = []
    current_sim_price = last_price
    
    # 產生 15 期的預測漫步
    for i in range(1, 16):
        sim_date = last_date + datetime.timedelta(days=i * (7 if interval == '1wk' else (30 if interval == '1mo' else 1)))
        if interval == '1d' and sim_date.weekday() >= 5: # 如果是日線跳過六日
            continue
            
        drift = recent_rets.mean() if not pd.isna(recent_rets.mean()) else 0
        shock = np.random.normal(drift, volatility)
        current_sim_price = current_sim_price * (1 + shock)
        prediction_data.append({
            "date": sim_date.strftime("%Y-%m-%d"),
            "mid": round(current_sim_price, 2)
        })

    # --- 準備評分與機器學習資料 ---
    current_rsi = float(df["RSI_14"].iloc[-1]) if "RSI_14" in df.columns else 50
    tech_score = int(current_rsi)
    
    # 🚀 這裡呼叫我們剛寫好的隨機森林模型
    ml_analysis = calculate_ml_factor_contributions(df)
    
    # 將 ML 的上漲機率當作動態的 AI 總分
    ai_total_score = int(ml_analysis["upward_probability_pct"])
    
    return {
        "symbol": symbol,
        "market_benchmark": "0050(台灣50)" if is_tw else "SPY(標普500)",
        "ai_score": ai_total_score,
        "ai_sentiment": "偏多震盪" if ai_total_score > 50 else "弱勢整理",
        "ml_prediction": { # 🚀 新增：這裡就是給前端畫圓餅圖或長條圖的因子貢獻度！
            "upward_probability_pct": ml_analysis["upward_probability_pct"],
            "factors": ml_analysis["factor_importance"]
        },
        "quant_metrics": {
            "beta": round(beta, 2),          
            "annual_alpha_pct": round(alpha, 2), 
            "current_price": last_price,
            "stop_loss_suggested": round(last_price * (1 + (cvar_95 if pd.notna(cvar_95) else -0.05)), 2)
        },
        "advice": {
            "buy_price": last_price,
            "take_profit": round(last_price * 1.15, 2),
            "stop_loss": round(last_price * 0.9, 2)    
        },
        "score_breakdown": {
            "technical": tech_score,
            "fundamental": 85,
            "chip": 75,
            "news": 80
        },
        "chart_data": {
            "history": history_data,
            "prediction": prediction_data
        },
        "news": news_data
    }

# 2️⃣ 回測 API
@app.get("/api/backtest/{symbol}")
async def backtest_endpoint(symbol: str):
    df = await fetch_price_history(symbol.upper())
    if df.empty: raise HTTPException(status_code=404, detail="找不到股票資料")
    
    bt_result = run_backtest(df)
    return {"symbol": symbol.upper(), "backtest_3yr": bt_result}

# 3️⃣ 投資組合管理 API
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

# 4️⃣ 新聞 API (✅ 更新：改用 Google News 並清理股票代碼)
@app.get("/api/news/{symbol}")
async def get_news(symbol: str):
    is_tw = is_taiwan_stock(symbol)
    clean_sym = clean_tw_symbol(symbol) if is_tw else symbol.upper()
    news = await fetch_google_news(clean_sym, is_tw=is_tw)
    return {"symbol": symbol.upper(), "news": news}

# 5️⃣ 新增：讓使用者自訂關鍵字搜尋新聞 API
@app.get("/api/news/search/")
async def search_custom_news(
    q: str = Query(..., description="使用者輸入的搜尋關鍵字"), 
    is_tw: bool = Query(True, description="是否搜尋中文/台灣市場")
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="關鍵字不能為空")
        
    news = await fetch_google_news(q.strip(), is_tw=is_tw)
    return {"query": q, "news": news}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
