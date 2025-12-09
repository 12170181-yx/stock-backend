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

# 允許跨域請求 (讓前端可以連線)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 設定快取：個股分析 30 分鐘，排行榜 1 小時
cache = TTLCache(maxsize=200, ttl=1800)
rank_cache = TTLCache(maxsize=1, ttl=3600)

class StockRequest(BaseModel):
    ticker: str
    principal: int = 100000
    risk: str = "neutral"

# --- 1. 精準技術指標運算 (Pandas/Numpy) ---
# 這是您的核心運算引擎，確保數據準確
def calculate_technicals(df):
    if len(df) < 60: return None
    
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

    # KD (簡易運算)
    low_min = low.rolling(window=9).min()
    high_max = high.rolling(window=9).max()
    rsv = ((close - low_min) / (high_max - low_min)) * 100
    k = rsv.ewm(com=2).mean() # 近似計算
    d = k.ewm(com=2).mean()

    # MA 均線
    ma5 = close.rolling(window=5).mean()
    ma20 = sma20
    ma60 = close.rolling(window=60).mean()

    # 安全數值轉換 (處理 NaN)
    def safe(series):
        val = series.iloc[-1]
        return float(val) if not np.isnan(val) else 0.0

    return {
        "rsi": safe(rsi),
        "macd": safe(macd),
        "upper": safe(upper),
        "lower": safe(lower),
        "k": safe(k),
        "d": safe(d),
        "ma5": safe(ma5),
        "ma20": safe(ma20),
        "ma60": safe(ma60),
        "price": safe(close)
    }

# --- 2. 籌碼面 (台股證交所爬蟲) ---
# 針對 .TW 結尾的股票抓取真實法人買賣超
def get_twse_chip(stock_code):
    if not (stock_code.isdigit() and len(stock_code) == 4): return 0
    try:
        # 抓取證交所最新的三大法人買賣超日報
        url = "https://www.twse.com.tw/rwd/zh/fund/T86?response=json&selectType=ALL"
        res = requests.get(url, timeout=4)
        if res.status_code == 200:
            data = res.json()
            if 'data' in data:
                for row in data['data']:
                    if row[0] == stock_code:
                        # 回傳三大法人合計買賣超股數 (處理千分位逗點)
                        return int(row[-1].replace(',', ''))
    except:
        pass
    return 0

# --- 3. 基本面 (Yahoo Info) ---
def get_fundamentals(info):
    score = 50
    try:
        # 根據 EPS, ROE, PE 給予評分
        if info.get('trailingEps', 0) > 0: score += 15
        if info.get('returnOnEquity', 0) > 0.1: score += 15
        
        pe = info.get('trailingPE', 0)
        if 0 < pe < 25: score += 10
        elif pe > 50: score -= 10 # 本益比過高扣分
        
        if info.get('profitMargins', 0) > 0.15: score += 10
    except:
        pass
    return min(99, max(1, score))

# --- 4. 真實新聞抓取 (New!) ---
def get_real_news(stock):
    try:
        news = stock.news
        result = []
        # 只取最新的 5 則新聞，並整理格式
        for n in news[:5]: 
            result.append({
                "title": n.get('title', '無標題'),
                "link": n.get('link', '#'),
                "publisher": n.get('publisher', 'Yahoo Finance'),
                "time": "Latest"
            })
        return result
    except:
        return []

# --- 5. 排行榜快速掃描 (New!) ---
def quick_scan(ticker):
    try:
        stock = yf.Ticker(ticker)
        # 只抓一個月數據做快速掃描
        hist = stock.history(period="1mo")
        if hist.empty: return None
        
        price = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[0]
        change = (price - prev) / prev
        
        # 簡單趨勢評分
        ma5 = hist['Close'].rolling(5).mean().iloc[-1]
        score = 60
        if price > ma5: score += 10
        if change > 0.05: score += 10
        elif change < -0.05: score -= 10
        
        return {
            "ticker": ticker,
            "price": round(price, 2),
            "change_pct": round(change * 100, 2),
            "score": min(99, max(1, score))
        }
    except:
        return None

# --- API 接口定義 ---

@app.get("/")
def home():
    return {"status": "ok", "version": "Integrated_Pro_v3"}

# 取得熱門股排行
@app.get("/rankings")
def get_rankings():
    # 檢查快取
    if "global_rank" in rank_cache:
        return rank_cache["global_rank"]
    
    # 預設掃描的熱門股清單 (含台股與美股)
    targets = ["2330.TW", "2317.TW", "2454.TW", "2603.TW", "NVDA", "AAPL", "TSLA", "AMD", "MSFT", "GOOG"]
    results = []
    
    for t in targets:
        res = quick_scan(t)
        if res: results.append(res)
    
    # 依分數高低排序
    results.sort(key=lambda x: x['score'], reverse=True)
    rank_cache["global_rank"] = results
    return results

# 個股深度分析
@app.post("/analyze")
def analyze(req: StockRequest):
    ticker = req.ticker.upper()
    
    # 檢查快取
    if ticker in cache:
        return cache[ticker]

    try:
        stock = yf.Ticker(ticker)
        # 抓取 1 年數據以確保長天期均線 (MA60) 能計算
        hist = stock.history(period="1y")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail="No Data")

        # A. 計算技術指標
        tech_data = calculate_technicals(hist)
        tech_score = 50
        if tech_data:
            # 根據 RSI 判斷
            if tech_data['rsi'] > 70: tech_score = 85
            elif tech_data['rsi'] < 30: tech_score = 25
            else: tech_score = 50 + (tech_data['rsi'] - 50) * 0.5
            
            # 根據均線判斷
            if tech_data['price'] > tech_data['ma20']: tech_score += 10
            if tech_data['ma5'] > tech_data['ma20']: tech_score += 10
            
            tech_score = min(99, max(1, int(tech_score)))

        # B. 計算基本面
        fund_score = get_fundamentals(stock.info)

        # C. 計算籌碼面 (台股專用)
        chip_val = 0
        if ".TW" in ticker:
            chip_val = get_twse_chip(ticker.split('.')[0])
        
        chip_score = 50
        if chip_val > 1000000: chip_score = 85
        elif chip_val > 0: chip_score = 60
        elif chip_val < -1000000: chip_score = 25
        elif chip_val < 0: chip_score = 40

        # D. 抓取真實新聞
        news_list = get_real_news(stock)
        news_score = 50 # 這裡保持中性，讓前端顯示新聞列表即可

        # E. 計算總分 (權威運算)
        # 權重分配：技術 40%, 基本 20%, 籌碼 20%, 消息 20%
        total_score = int(tech_score * 0.4 + fund_score * 0.2 + chip_score * 0.2 + news_score * 0.2)

        # 準備前端需要的圖表數據
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()
        
        # 簡單趨勢預測線 (未來 5 天)
        last_price = prices[-1]
        trend = 1.002 if tech_score > 60 else 0.998
        future_dates = []
        future_means = []
        last_date_obj = datetime.datetime.strptime(dates[-1], '%Y-%m-%d')
        
        for i in range(1, 6):
            next_d = last_date_obj + datetime.timedelta(days=i)
            future_dates.append(next_d.strftime('%Y-%m-%d'))
            future_means.append(round(last_price * (trend ** i), 2))

        # 最終回傳結構
        result = {
            "ticker": ticker,
            "current_price": round(last_price, 2),
            "total_score": total_score,
            "evaluation": "多頭" if total_score > 65 else "觀望",
            "recommendation": "買進" if total_score > 70 else "持有",
            "details": {
                "tech": tech_score,
                "fund": fund_score,
                "chip": chip_score,
                "news": news_score
            },
            # 傳回伺服器算好的精準指標
            "tech_indicators": tech_data,
            # 傳回真實新聞列表
            "news_list": news_list,
            # 傳回圖表數據
            "chart_data": {
                "history_date": dates,
                "history_price": [round(p, 2) for p in prices],
                "future_date": future_dates,
                "future_mean": future_means
            }
        }
        
        cache[ticker] = result
        return result

    except Exception as e:
        print(f"Server Error: {e}")
        raise HTTPException(status_code=500, detail="Analysis Failed")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

