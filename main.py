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
from sklearn.ensemble import RandomForestRegressor # 🌟 用於計算因子貢獻度

APP_NAME = "stock-backend-quant-pro"
DATABASE_PATH = os.getenv("DATABASE_PATH", "stock_app.db")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
FMP_API_KEY = os.getenv("FMP_API_KEY", "demo")

PRICE_CACHE: Dict[str, Tuple[pd.DataFrame, float]] = {}
FUND_CACHE: Dict[str, Tuple[Dict[str, Any], float]] = {}
NEWS_CACHE: Dict[str, Tuple[List[Dict[str, Any]], float]] = {}

PRICE_CACHE_TTL = 600       
FUND_CACHE_TTL = 86400     
NEWS_CACHE_TTL = 3600

# ==========================
# ⚙️ FastAPI 初始化 & DB (保持原樣)
# ==========================
async def init_db():
    async with aiosqlite.connect(DATABASE_PATH) as db:
        await db.execute("PRAGMA journal_mode=WAL;")
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class AnalysisRequest(BaseModel):
    symbol: str
    principal: float = 100000
    duration: str = "mid"
    interval: str = "1d"

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
    df.ta.sma(length=200, append=True) # 🌟 新增 200MA，用於判斷牛熊市 Regime
    df.ta.ema(length=60, append=True) 
    df.ta.macd(append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True) # 新增布林通道 (用於因子特徵)
    df["volatility_20"] = df["ret"].rolling(20).std() * np.sqrt(252)
    return df.dropna(subset=["SMA_20", "volatility_20", "MACD_12_26_9", "SMA_200"])

# (保留你的 fetch_google_news 邏輯，為節省版面稍微縮寫)
async def fetch_google_news(keyword: str, is_tw: bool = True) -> List[Dict[str, Any]]:
    # ... 原本的 google news 抓取邏輯 ...
    return [{"title": f"模擬新聞: {keyword} 最新動態", "link": "", "published": "2024-01-01 12:00"}]

async def fetch_price_history(symbol: str, interval: str = "1d") -> pd.DataFrame:
    # ... 保留原本的 finmind / FMP 抓取邏輯 ...
    # 這裡為了展示完整架構，略寫原始碼，請維持你原有的 fetch_price_history 實作，確保 return enrich_indicators(df)
    pass # ⚠️ 記得把你的 fetch_price_history 貼回來這裡

async def fetch_benchmark(is_tw: bool) -> pd.DataFrame:
    bench_symbol = "0050" if is_tw else "SPY"
    return await fetch_price_history(bench_symbol)

# ==========================
# 🌟 [核心模組 1] 因子貢獻透明化 (Explainability)
# ==========================
def calculate_feature_importance(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """使用 Random Forest 評估技術指標對次日報酬的貢獻度 (模擬 SHAP)"""
    # 準備特徵 X 與目標 Y (次日報酬)
    features = ["SMA_20", "EMA_60", "MACD_12_26_9", "RSI_14", "BBL_20_2.0", "BBU_20_2.0", "volatility_20"]
    ml_df = df[features + ["ret"]].copy().dropna()
    ml_df["target"] = ml_df["ret"].shift(-1) # 預測明天
    ml_df = ml_df.dropna()

    if len(ml_df) < 50:
        return []

    X = ml_df[features]
    y = ml_df["target"]

    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=3)
    model.fit(X, y)

    importances = model.feature_importances_
    
    # 轉化為前端需要的格式，並加上正負向影響判斷 (用相關係數輔助)
    correlations = X.corrwith(y)
    shap_data = []
    
    feature_names_zh = {
        "SMA_20": "短期均線支撐", "EMA_60": "中期趨勢", "MACD_12_26_9": "MACD 動能",
        "RSI_14": "RSI 超買超賣", "BBL_20_2.0": "下軌乖離", "BBU_20_2.0": "上軌壓力", "volatility_20": "波動率風險"
    }

    for idx, col in enumerate(features):
        direction = "positive" if correlations[col] > 0 else "negative"
        # 為了視覺效果，把重要性轉為 1~100 的相對百分比
        value = round(importances[idx] * 100, 1)
        if value > 2: # 只傳遞貢獻度大於 2% 的因子
            shap_data.append({
                "factor": feature_names_zh.get(col, col),
                "value": value if direction == "positive" else -value,
                "fill": "#ef4444" if direction == "positive" else "#22c55e"
            })
            
    # 依絕對值排序
    shap_data.sort(key=lambda x: abs(x["value"]), reverse=True)
    return shap_data

# ==========================
# 📈 回測與績效計算引擎
# ==========================
def run_backtest(df: pd.DataFrame) -> Dict[str, Any]:
    df["signal"] = np.where((df["MACDh_12_26_9"] > 0) & (df["Close"] > df["EMA_60"]), 1, 0)
    df["strategy_ret"] = df["signal"].shift(1) * df["ret"]
    valid_rets = df["strategy_ret"].dropna()
    
    if valid_rets.empty: return {}

    cum_ret = (1 + valid_rets).cumprod().iloc[-1] - 1
    
    # 基本指標
    mdd = ( (1+valid_rets).cumprod() / (1+valid_rets).cumprod().cummax() - 1 ).min() * 100
    sharpe = (valid_rets.mean() / valid_rets.std()) * np.sqrt(252) if valid_rets.std() > 0 else 0

    # 🌟 [核心模組 2] 穩健性：Rolling Window Test (滾動 252 天夏普值)
    rolling_sharpe = valid_rets.rolling(252).apply(lambda x: (x.mean() / x.std()) * np.sqrt(252) if x.std() > 0 else 0).dropna()
    rolling_sharpe_data = [
        {"month": str(date)[:7], "sharpe": round(val, 2)} 
        for date, val in rolling_sharpe.resample('M').last().items()
    ]

    # 🌟 [核心模組 2] 穩健性：Regime Analysis (牛熊市分段)
    # 定義 Regime：收盤價 > 200MA 為牛市，反之為熊市
    df["regime"] = np.where(df["Close"] > df["SMA_200"], "Bull", "Bear")
    bull_rets = df[df["regime"] == "Bull"]["strategy_ret"].dropna()
    bear_rets = df[df["regime"] == "Bear"]["strategy_ret"].dropna()
    
    bull_win = (bull_rets > 0).mean() * 100 if len(bull_rets) > 0 else 0
    bear_win = (bear_rets > 0).mean() * 100 if len(bear_rets) > 0 else 0
    bull_ret = ((1 + bull_rets).prod() - 1) * 100 if len(bull_rets) > 0 else 0
    bear_ret = ((1 + bear_rets).prod() - 1) * 100 if len(bear_rets) > 0 else 0

    regime_data = [
        {"regime": "大盤多頭 (200MA之上)", "winRate": round(bull_win, 1), "return": round(bull_ret, 1)},
        {"regime": "大盤空頭 (200MA之下)", "winRate": round(bear_win, 1), "return": round(bear_ret, 1)}
    ]

    return {
        "cumulative_return_pct": round(cum_ret * 100, 2),
        "max_drawdown_pct": round(abs(mdd), 2),
        "sharpe_ratio": round(sharpe, 2),
        "robustness": {
            "rolling_sharpe": rolling_sharpe_data[-24:], # 取近兩年
            "regime_analysis": regime_data
        }
    }

# ==========================
# 🚀 API 路由區
# ==========================

# 1️⃣ 分析與因子拆解 API 
@app.post("/api/analyze")
async def analyze(request: AnalysisRequest):
    df = await fetch_price_history(request.symbol.upper(), request.interval)
    if df.empty: raise HTTPException(status_code=404, detail="資料不足")

    last_price = float(df["Close"].iloc[-1])
    recent_rets = df["ret"].tail(252).dropna()
    
    # 🌟 [核心模組 4] 風險模型 (Risk Modeling): VaR & CVaR & Volatility
    # 使用歷史模擬法計算 95% 信心水準的風險價值 (VaR)
    var_95 = np.percentile(recent_rets, 5) 
    cvar_95 = recent_rets[recent_rets <= var_95].mean()
    annual_volatility = recent_rets.std() * np.sqrt(252)

    # 取得特徵貢獻度
    shap_data = calculate_feature_importance(df)

    return {
        "symbol": request.symbol.upper(),
        "ai_score": 85, 
        "ai_sentiment": "偏多震盪" if float(df["RSI_14"].iloc[-1]) > 50 else "弱勢整理",
        "risk_metrics": { # ✅ 提供給前端的高階風險數據
            "beta": 1.15, # 簡化寫死，真實需用 fetch_benchmark 計算 Covariance
            "volatility": round(annual_volatility * 100, 2),
            "var95": round(abs(var_95) * 100, 2),
            "cvar95": round(abs(cvar_95) * 100, 2)
        },
        "score_breakdown": {"technical": 88, "fundamental": 85, "chip": 75, "news": 80},
        "feature_importance": shap_data, # ✅ 因子貢獻透明化資料
        "chart_data": {"history": [], "prediction": []}, # 保留你的圖表組合邏輯
        "advice": {"buy_price": last_price, "take_profit": round(last_price * 1.15, 2), "stop_loss": round(last_price * (1+var_95), 2)}
    }

# 2️⃣ 回測與穩健性 API
@app.get("/api/backtest/{symbol}")
async def backtest_endpoint(symbol: str):
    df = await fetch_price_history(symbol.upper())
    if df.empty: raise HTTPException(status_code=404, detail="找不到股票資料")
    bt_result = run_backtest(df)
    return {"symbol": symbol.upper(), "backtest": bt_result}

# 🌟 [核心模組 3] 橫向比較 (Cross-sectional Ranking) API
@app.get("/api/ranking")
async def market_ranking():
    """
    全市場掃描器 API。
    實務上應由背景排程(Celery/Cron)將全市場股票算好存入 DB，前端直接讀 DB。
    此處用預設的「台股權值股清單」做即時動態計算展示。
    """
    target_symbols = ["2330.TW", "2317.TW", "2454.TW", "2382.TW", "2881.TW", "2603.TW", "3231.TW"]
    ranking_results = []

    for sym in target_symbols:
        # 實務上這裡該讀 DB，為了展示我們直接抓取 (如果有 cache 很快)
        df = await fetch_price_history(sym, "1d")
        if df.empty: continue
        
        last_row = df.iloc[-1]
        
        # 簡單的多因子評分模型 (Momentum + Volatility)
        rsi = last_row.get("RSI_14", 50)
        macd_h = last_row.get("MACDh_12_26_9", 0)
        price_to_200ma = last_row["Close"] / last_row.get("SMA_200", last_row["Close"])
        
        momentum_score = min(max((rsi - 30) * 2, 0), 100) # RSI 轉 0-100 分
        trend_score = 100 if price_to_200ma > 1.05 else (50 if price_to_200ma > 0.95 else 20)
        
        total_score = int(momentum_score * 0.6 + trend_score * 0.4)
        
        signal = "強烈買進" if total_score > 80 and macd_h > 0 else ("買進" if total_score > 60 else "觀望")

        ranking_results.append({
            "symbol": sym,
            "name": sym, # 實務上可串接股票名稱對照表
            "sector": "大型權值",
            "score": total_score,
            "momentum": int(momentum_score),
            "value": int(trend_score), # 暫代 Value 因子
            "signal": signal
        })

    # 依總分降序排序，並給予 Rank
    ranking_results.sort(key=lambda x: x["score"], reverse=True)
    for i, res in enumerate(ranking_results):
        res["rank"] = i + 1

    return {"data": ranking_results}

# 4️⃣ 投資組合與新聞 API (維持原樣)
# ... 請保留你原有的 @app.post("/api/portfolio") 與 @app.get("/api/news/{symbol}") ...

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
