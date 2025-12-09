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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 快取：1小時內同一支股票，保證回傳完全一樣的 JSON
cache = TTLCache(maxsize=200, ttl=3600)

class StockRequest(BaseModel):
    ticker: str
    principal: int = 100000
    risk: str = "neutral"

# --- 1. 精準技術指標 (Pandas) ---
def calculate_technicals(df):
    if len(df) < 30: return None
    close = df['Close']
    
    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    upper = sma20 + (std20 * 2)
    lower = sma20 - (std20 * 2)

    # KD (簡易)
    low_min = df['Low'].rolling(window=9).min()
    high_max = df['High'].rolling(window=9).max()
    rsv = ((close - low_min) / (high_max - low_min)) * 100
    k = rsv.ewm(com=2).mean()
    
    # MA
    ma5 = close.rolling(window=5).mean()
    ma20 = sma20
    ma60 = close.rolling(window=60).mean()

    def safe(series):
        val = series.iloc[-1]
        return float(val) if not np.isnan(val) else 0.0

    return {
        "rsi": safe(rsi),
        "macd": safe(macd),
        "upper": safe(upper),
        "lower": safe(lower),
        "k": safe(k),
        "ma5": safe(ma5),
        "ma20": safe(ma20),
        "ma60": safe(ma60),
        "price": safe(close)
    }

# --- 2. 籌碼面 (台股) ---
def get_twse_chip(stock_code):
    if not (stock_code.isdigit() and len(stock_code) == 4): return 0
    try:
        url = "https://www.twse.com.tw/rwd/zh/fund/T86?response=json&selectType=ALL"
        res = requests.get(url, timeout=3)
        if res.status_code == 200:
            data = res.json()
            if 'data' in data:
                for row in data['data']:
                    if row[0] == stock_code:
                        return int(row[-1].replace(',', ''))
    except:
        pass
    return 0

# --- 3. 基本面 ---
def get_fundamentals(info):
    score = 50
    try:
        if info.get('trailingEps', 0) > 0: score += 15
        if info.get('returnOnEquity', 0) > 0.1: score += 15
        if 0 < info.get('trailingPE', 0) < 25: score += 10
        if info.get('profitMargins', 0) > 0.15: score += 10
    except:
        pass
    return min(99, max(1, score))

@app.post("/analyze")
def analyze(req: StockRequest):
    ticker = req.ticker.upper()
    if ticker in cache: return cache[ticker]

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="6mo")
        if hist.empty: raise HTTPException(status_code=404, detail="No Data")

        # A. 計算技術指標與分數
        tech = calculate_technicals(hist)
        tech_score = 50
        if tech:
            if tech['rsi'] > 70: tech_score = 85
            elif tech['rsi'] < 30: tech_score = 25
            else: tech_score = 50 + (tech['rsi'] - 50) * 0.5
            
            if tech['price'] > tech['ma20']: tech_score += 10
            if tech['ma5'] > tech['ma20']: tech_score += 10
            tech_score = min(99, max(1, int(tech_score)))

        # B. 計算基本面
        fund_score = get_fundamentals(stock.info)

        # C. 計算籌碼面
        chip_val = 0
        if ".TW" in ticker:
            chip_val = get_twse_chip(ticker.split('.')[0])
        
        chip_score = 50
        if chip_val > 1000000: chip_score = 90
        elif chip_val > 0: chip_score = 65
        elif chip_val < -1000000: chip_score = 20
        elif chip_val < 0: chip_score = 40

        # D. 消息面 (暫時中性)
        news_score = 50

        # E. 總分 (由後端一錘定音)
        total_score = int(tech_score * 0.4 + fund_score * 0.2 + chip_score * 0.2 + news_score * 0.2)

        # 回傳完整資料 (包含圖表)
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()
        
        # 預測線
        last_p = prices[-1]
        trend = 1.005 if tech and tech['ma5'] > tech['ma20'] else 0.995
        f_dates = [(datetime.datetime.strptime(dates[-1], '%Y-%m-%d') + datetime.timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1,6)]
        f_means = [round(last_p * (trend ** i), 2) for i in range(1,6)]

        result = {
            "ticker": ticker,
            "current_price": round(last_p, 2),
            "total_score": total_score,  # 這是唯一的真理分數
            "evaluation": "多頭" if total_score > 60 else "空頭",
            "scores": {
                "tech": tech_score,
                "fund": fund_score,
                "chip": chip_score,
                "news": news_score
            },
            "tech_details": tech, # 用於前端顯示詳細數字，但不參與前端運算
            "chart_data": {
                "history_date": dates,
                "history_price": [round(p, 2) for p in prices],
                "future_date": f_dates,
                "future_mean": f_means,
                # 為節省頻寬，只傳必要數據
                "future_upper": [round(p*1.05, 2) for p in f_means],
                "future_lower": [round(p*0.95, 2) for p in f_means]
            }
        }
        
        cache[ticker] = result
        return result

    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
