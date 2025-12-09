import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
from cachetools import TTLCache

app = FastAPI()

# 允許跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 伺服器端快取：確保短時間內所有人查到的分數完全一樣
cache = TTLCache(maxsize=200, ttl=1800) # 30分鐘快取

class StockRequest(BaseModel):
    ticker: str
    principal: int = 100000
    risk: str = "neutral"

# --- 1. [權威運算] 伺服器端技術指標 ---
# 這裡算出來的數字是最終標準，前端不准重算
def calculate_server_technicals(df):
    # 確保資料足夠，否則不計算
    if len(df) < 120: 
        return None
    
    close = df['Close']
    
    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    # MA (5, 20, 60)
    ma5 = close.rolling(window=5).mean().iloc[-1]
    ma20 = close.rolling(window=20).mean().iloc[-1]
    ma60 = close.rolling(window=60).mean().iloc[-1]

    # MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_val = macd.iloc[-1]

    # KD (9, 3, 3) - 簡易版
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = ((close - low_min) / (high_max - low_min)) * 100
    k = rsv.ewm(com=2).mean().iloc[-1]
    d = k.ewm(com=2).mean().iloc[-1]

    # 布林通道
    std20 = close.rolling(window=20).std().iloc[-1]
    upper = ma20 + (std20 * 2)
    lower = ma20 - (std20 * 2)

    # --- [評分邏輯] 伺服器端統一標準 ---
    score = 50
    
    # RSI 評分
    if current_rsi > 70: score += 15 # 強勢
    elif current_rsi < 30: score -= 10 # 弱勢
    else: score += (current_rsi - 50) * 0.5

    # 均線評分 (多頭排列)
    price = close.iloc[-1]
    if price > ma5 > ma20 > ma60: score += 15
    elif price < ma5 < ma20 < ma60: score -= 15
    elif price > ma20: score += 5

    # MACD 評分
    if macd_val > 0: score += 5

    return {
        "score": min(99, max(1, int(score))),
        "indicators": {
            "rsi": round(float(current_rsi), 2),
            "ma20": round(float(ma20), 2),
            "macd": round(float(macd_val), 2),
            "k": round(float(k), 2),
            "upper": round(float(upper), 2),
            "lower": round(float(lower), 2)
        }
    }

# --- 2. [權威數據] 證交所/Yahoo ---
def get_fundamentals_score(info):
    score = 50
    try:
        if info.get('trailingEps', 0) > 0: score += 10
        if info.get('returnOnEquity', 0) > 0.1: score += 10
        if 0 < info.get('trailingPE', 0) < 25: score += 10
    except: pass
    return score

def get_chip_score(stock_code):
    # 台股籌碼邏輯 (簡化版，確保穩定回傳)
    # 這裡如果爬蟲失敗，會回傳 50 分中性，保證不報錯
    if not (stock_code.isdigit() and len(stock_code) == 4): return 50
    try:
        url = "https://www.twse.com.tw/rwd/zh/fund/T86?response=json&selectType=ALL"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            for row in data.get('data', []):
                if row[0] == stock_code:
                    val = float(row[-1].replace(',', ''))
                    if val > 1000: return 80
                    if val < -1000: return 30
                    return 60
    except: pass
    return 50

@app.post("/analyze")
def analyze(req: StockRequest):
    ticker = req.ticker.upper()
    
    if ticker in cache:
        return cache[ticker]

    try:
        stock = yf.Ticker(ticker)
        # 強制抓取 1 年數據，確保技術指標運算基底一致
        hist = stock.history(period="1y")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="查無資料")

        # 1. 計算技術面 (Technical) - 40%
        tech_result = calculate_server_technicals(hist)
        tech_score = tech_result['score'] if tech_result else 50
        
        # 2. 計算基本面 (Fundamental) - 30%
        fund_score = get_fundamentals_score(stock.info)
        
        # 3. 計算籌碼面 (Chip) - 15%
        chip_score = 50
        if ".TW" in ticker:
            chip_score = get_chip_score(ticker.split('.')[0])
        
        # 4. 消息面 (News) - 15%
        news_score = 50 # 暫時中性

        # 5. [最終總分] 伺服器蓋章認證
        total_score = int(tech_score * 0.4 + fund_score * 0.3 + chip_score * 0.15 + news_score * 0.15)

        # 準備圖表數據
        chart_dates = hist.index.strftime('%Y-%m-%d').tolist()
        chart_prices = hist['Close'].tolist()
        
        # 簡單預測
        future_dates = []
        future_means = []
        last_price = chart_prices[-1]
        last_date = datetime.datetime.strptime(chart_dates[-1], '%Y-%m-%d')
        trend = 1.002 if tech_score > 60 else 0.998
        for i in range(1, 6):
            future_dates.append((last_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d'))
            future_means.append(round(last_price * (trend ** i), 2))

        response = {
            "ticker": ticker,
            "current_price": round(last_price, 2),
            "total_score": total_score,  # 這是唯一的真理分數
            "evaluation": "強勢多頭" if total_score > 75 else ("區間整理" if total_score > 60 else "弱勢"),
            "recommendation": "買進" if total_score > 70 else "觀望",
            "details": {
                "tech": tech_score,
                "fund": fund_score,
                "chip": chip_score,
                "news": news_score
            },
            # 附上伺服器算好的精確指標，前端直接顯示即可
            "tech_indicators": tech_result['indicators'] if tech_result else {},
            "chart_data": {
                "history_date": chart_dates,
                "history_price": [round(p, 2) for p in chart_prices],
                "future_date": future_dates,
                "future_mean": future_means
            }
        }
        
        cache[ticker] = response
        return response

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="分析失敗")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
