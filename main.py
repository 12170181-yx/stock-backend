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

# 允許跨域請求 (讓您的 Vercel 前端可以連線)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 快取設定：TTL 3600秒 (1小時)，避免頻繁爬取導致 IP 被鎖
cache = TTLCache(maxsize=200, ttl=3600)

class StockRequest(BaseModel):
    ticker: str
    principal: int = 100000
    risk: str = "neutral"

# --- 1. 專業技術指標運算 (Pandas) ---
def calculate_technicals(df):
    if len(df) < 60:
        return None
    
    # 收盤價序列
    close = df['Close']
    high = df['High']
    low = df['Low']

    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    signal = macd.ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20, 2)
    sma20 = close.rolling(window=20).mean()
    std20 = close.rolling(window=20).std()
    upper = sma20 + (std20 * 2)
    lower = sma20 - (std20 * 2)

    # KD (9, 3, 3) - 簡易版
    low_min = low.rolling(window=9).min()
    high_max = high.rolling(window=9).max()
    rsv = ((close - low_min) / (high_max - low_min)) * 100
    k = rsv.ewm(com=2).mean() # 近似計算
    d = k.ewm(com=2).mean()

    # 確保回傳數值安全 (處理 NaN)
    def safe_val(series):
        val = series.iloc[-1]
        return float(val) if not np.isnan(val) else 0.0

    return {
        "rsi": safe_val(rsi),
        "macd": safe_val(macd),
        "signal": safe_val(signal),
        "upper": safe_val(upper),
        "lower": safe_val(lower),
        "k": safe_val(k),
        "d": safe_val(d),
        "ma5": safe_val(close.rolling(window=5).mean()),
        "ma20": safe_val(sma20),
        "ma60": safe_val(close.rolling(window=60).mean())
    }

# --- 2. 爬取證交所籌碼 (TWSE) ---
def get_twse_chip(stock_code):
    # 僅針對台股 4 碼數字代碼
    if not (stock_code.isdigit() and len(stock_code) == 4):
        return 0
    
    try:
        # 嘗試抓取最新的三大法人買賣超 (這是公開接口)
        url = f"https://www.twse.com.tw/rwd/zh/fund/T86?response=json&selectType=ALL"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=4)
        
        if res.status_code == 200:
            data = res.json()
            if 'data' in data:
                for row in data['data']:
                    if row[0] == stock_code:
                        # 這是「三大法人合計買賣超股數」
                        # 格式通常包含逗號，需處理
                        val_str = row[-1].replace(',', '')
                        return int(val_str)
    except:
        pass # 爬蟲失敗時回傳 0，不讓程式崩潰
    return 0

# --- 3. 獲取基本面 (Yahoo) ---
def get_fundamentals(info):
    try:
        # 嘗試從 info 字典中提取關鍵數據
        # 若無數據則回傳 0
        eps = info.get('trailingEps', 0)
        pe = info.get('trailingPE', 0)
        roe = info.get('returnOnEquity', 0)
        profit_margin = info.get('profitMargins', 0)
        
        # 簡單評分轉化 (0-100)
        score = 50
        if eps and eps > 0: score += 10
        if roe and roe > 0.1: score += 10
        if profit_margin and profit_margin > 0.1: score += 10
        if pe and pe > 0 and pe < 25: score += 10
        if pe and pe > 40: score -= 10
        
        return max(1, min(99, score))
    except:
        return 50

@app.get("/")
def home():
    return {"status": "ok", "version": "Pro_v1.0"}

@app.post("/analyze")
def analyze(req: StockRequest):
    ticker = req.ticker.upper()
    
    # 檢查快取
    if ticker in cache:
        return cache[ticker]

    try:
        # 1. 抓取 Yahoo 數據
        stock = yf.Ticker(ticker)
        # 抓 1 年數據以確保有足夠樣本計算 MA60, MA120
        hist = stock.history(period="1y") 
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="查無此股")

        # 2. 執行技術指標運算
        tech_data = calculate_technicals(hist)
        
        # 3. 抓取基本面
        fund_score = get_fundamentals(stock.info)
        
        # 4. 抓取籌碼面 (如果是台股)
        chip_val = 0
        if ".TW" in ticker:
            code = ticker.split('.')[0]
            chip_val = get_twse_chip(code)
        
        # 籌碼評分邏輯
        chip_score = 50
        if chip_val > 1000000: chip_score = 85 # 大買
        elif chip_val > 0: chip_score = 60
        elif chip_val < -1000000: chip_score = 25 # 大賣
        elif chip_val < 0: chip_score = 40

        # 5. 消息面 (模擬，因 Google News API 付費且貴)
        news_score = 50 

        # 6. 建構回應格式 (配合前端)
        current_price = hist['Close'].iloc[-1]
        
        # 準備圖表數據
        chart_dates = hist.index.strftime('%Y-%m-%d').tolist()
        chart_prices = hist['Close'].tolist()
        
        # 簡單預測線 (未來 5 天)
        future_dates = []
        future_means = []
        future_uppers = []
        future_lowers = []
        last_date = datetime.datetime.strptime(chart_dates[-1], '%Y-%m-%d')
        
        trend = 1.005 if tech_data and tech_data['ma5'] > tech_data['ma20'] else 0.995
        
        for i in range(1, 6):
            next_d = last_date + datetime.timedelta(days=i)
            future_dates.append(next_d.strftime('%Y-%m-%d'))
            pred = current_price * (trend ** i)
            future_means.append(round(pred, 2))
            future_uppers.append(round(pred * 1.05, 2))
            future_lowers.append(round(pred * 0.95, 2))

        response_data = {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "recommendation": "買進" if tech_data and tech_data['rsi'] < 30 else "持有",
            "details": {
                "fund": fund_score,
                "chip": chip_score,
                "news": news_score,
                # 技術面交給前端算，或後端算好傳過去也可，這裡示範後端傳基礎值
                "tech_rsi": tech_data['rsi'] if tech_data else 50
            },
            "chart_data": {
                "history_date": chart_dates,
                "history_price": [round(p, 2) for p in chart_prices],
                "future_date": future_dates,
                "future_mean": future_means,
                "future_upper": future_uppers,
                "future_lower": future_lowers
            }
        }
        
        # 寫入快取
        cache[ticker] = response_data
        return response_data

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="後端運算錯誤")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
