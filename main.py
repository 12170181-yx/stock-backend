import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
from cachetools import TTLCache

app = FastAPI(title="AI Stock Analysis API - Public Mode")

# --- CORS 設定 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 模擬資料庫 (公開測試用) ---
# 所有人都共用 "demo_user" 的資料，方便測試
demo_portfolio = [] 
demo_favorites = []

# --- 快取設定 ---
cache = TTLCache(maxsize=200, ttl=1800) # 30分鐘
rank_cache = TTLCache(maxsize=1, ttl=3600) # 1小時

# --- 資料模型 ---
class PortfolioItem(BaseModel):
    symbol: str
    cost_price: float
    shares: int
    date: str

class AnalyzeRequest(BaseModel):
    symbol: str
    principal: int = 100000
    risk: str = "neutral"

# --- 核心演算法 ---
def calculate_technicals(df):
    if len(df) < 60: return None
    close = df['Close']
    
    # RSI (14)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # MACD
    exp12 = close.ewm(span=12, adjust=False).mean()
    exp26 = close.ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    
    # Bollinger
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + (std20 * 2)
    lower = sma20 - (std20 * 2)

    # KD
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = ((close - low_min) / (high_max - low_min)) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()

    # MA
    ma5 = close.rolling(5).mean()
    ma20 = sma20
    ma60 = close.rolling(60).mean()

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

def get_twse_chip(stock_code):
    if not (stock_code.isdigit() and len(stock_code) == 4): return 0
    try:
        url = "https://www.twse.com.tw/rwd/zh/fund/T86?response=json&selectType=ALL"
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=4)
        if res.status_code == 200:
            data = res.json()
            if 'data' in data:
                for row in data['data']:
                    if row[0] == stock_code:
                        return int(row[-1].replace(',', ''))
    except: pass
    return 0

def get_real_news(stock):
    try:
        news = stock.news
        result = []
        for n in news[:5]:
            result.append({
                "title": n.get('title'),
                "link": n.get('link'),
                "publisher": n.get('publisher', 'Yahoo')
            })
        return result
    except: return []

def detect_patterns(df):
    patterns = []
    if len(df) < 5: return patterns
    last = df.iloc[-1]
    body = abs(last['Close'] - last['Open'])
    upper = last['High'] - max(last['Close'], last['Open'])
    lower = min(last['Close'], last['Open']) - last['Low']
    
    if lower > body * 2: patterns.append("錘子線")
    if last['Close'] > last['Open'] and body > (last['High']-last['Low'])*0.8: patterns.append("長紅K")
    
    ma5 = df['Close'].rolling(5).mean()
    ma20 = df['Close'].rolling(20).mean()
    if ma5.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-2] <= ma20.iloc[-2]: patterns.append("MA金叉")
    
    return patterns

# --- API Endpoints (無須驗證) ---

@app.get("/")
def home():
    return {"status": "Public_Backend_Ready"}

# 1. 深度分析
@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    ticker = req.symbol.upper()
    
    if ticker in cache: return cache[ticker]

    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty: raise HTTPException(status_code=404, detail="No Data")

        # 技術面
        tech_data = calculate_technicals(hist)
        tech_score = 50
        if tech_data:
            if tech_data['rsi'] > 70: tech_score = 85
            elif tech_data['rsi'] < 30: tech_score = 25
            else: tech_score = 50 + (tech_data['rsi'] - 50) * 0.5
            
            if tech_data['price'] > tech_data['ma20']: tech_score += 10
            if tech_data['ma5'] > tech_data['ma20']: tech_score += 10
        tech_score = min(99, max(1, int(tech_score)))

        # 基本面
        info = stock.info
        fund_score = 50
        if info.get('trailingEps', 0) > 0: fund_score += 20
        if info.get('returnOnEquity', 0) > 0.1: fund_score += 20
        pe = info.get('trailingPE', 0)
        if 0 < pe < 25: fund_score += 10
        fund_score = min(99, max(1, fund_score))

        # 籌碼面
        chip_val = 0
        if ".TW" in ticker:
            chip_val = get_twse_chip(ticker.split('.')[0])
        chip_score = 50
        if chip_val > 1000000: chip_score = 85
        elif chip_val > 0: chip_score = 60
        elif chip_val < -1000000: chip_score = 25
        elif chip_val < 0: chip_score = 40

        # 新聞與總分
        news_list = get_real_news(stock)
        news_score = 50
        total_score = int(tech_score*0.4 + fund_score*0.2 + chip_score*0.2 + news_score*0.2)

        # 其他數據
        returns = hist['Close'].pct_change().dropna()
        var_95 = returns.quantile(0.05) if len(returns) > 0 else 0
        max_loss = req.principal * abs(var_95)

        high_low = hist['High'] - hist['Low']
        atr = high_low.rolling(14).mean().iloc[-1]
        price = hist['Close'].iloc[-1]
        
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()
        
        # 預測線
        future_dates = []
        future_means = []
        last_date = datetime.datetime.strptime(dates[-1], '%Y-%m-%d')
        trend = 1.002 if tech_score > 60 else 0.998
        for i in range(1, 6):
            future_dates.append((last_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d'))
            future_means.append(round(price * (trend ** i), 2))

        result = {
            "symbol": ticker,
            "current_price": round(price, 2),
            "ai_score": total_score,
            "evaluation": "多頭" if total_score > 60 else "空頭",
            "recommendation": "買進" if total_score > 70 else "持有",
            "scores": {"tech": tech_score, "fund": fund_score, "chip": chip_score, "news": news_score},
            "roi": {
                "1d": round(returns.iloc[-1]*100, 2),
                "1w": round(hist['Close'].pct_change(5).iloc[-1]*100, 2),
                "1m": round(hist['Close'].pct_change(20).iloc[-1]*100, 2),
                "1y": round(hist['Close'].pct_change(250).iloc[-1]*100, 2)
            },
            "risk": {
                "var_95_pct": round(var_95*100, 2),
                "max_loss_est": round(max_loss, 0)
            },
            "strategy": {
                "entry": round(price, 2),
                "stop_loss": round(price - atr*2, 2),
                "take_profit": round(price + atr*3, 2)
            },
            "patterns": detect_patterns(hist),
            "tech_details": tech_data,
            "news_list": news_list,
            "chart_data": {
                "history_date": dates,
                "history_price": [round(p, 2) for p in prices],
                "future_date": future_dates,
                "future_mean": future_means
            },
            "totalScore": total_score,
            "currentPrice": round(price, 2),
            "details": {"tech": tech_score, "fund": fund_score, "chip": chip_score, "news": news_score},
            "tech_indicators": tech_data
        }
        
        cache[ticker] = result
        return result

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="分析失敗")

# 2. 排行榜
@app.get("/api/rankings")
def get_rankings():
    if "global_rank" in rank_cache: return rank_cache["global_rank"]
    targets = ["2330.TW", "NVDA", "AAPL", "2317.TW", "TSLA", "AMD", "MSFT"]
    results = []
    for t in targets:
        try:
            s = yf.Ticker(t)
            h = s.history(period="5d")
            if not h.empty:
                p = h['Close'].iloc[-1]
                chg = (p - h['Close'].iloc[0]) / h['Close'].iloc[0]
                results.append({"ticker": t, "price": round(p,2), "change_pct": round(chg*100,2), "score": 80 if chg>0 else 45})
        except: pass
    rank_cache["global_rank"] = results
    return results

# 3. 模擬資產 & 收藏 (公開版)
@app.get("/api/favorites")
def get_favorites(): return demo_favorites

@app.post("/api/favorites/{symbol}")
def add_favorite(symbol: str):
    if symbol not in demo_favorites: demo_favorites.append(symbol)
    return demo_favorites

@app.delete("/api/favorites/{symbol}")
def remove_favorite(symbol: str):
    if symbol in demo_favorites: demo_favorites.remove(symbol)
    return demo_favorites

@app.get("/api/portfolio")
def get_portfolio(): return demo_portfolio

@app.post("/api/portfolio/add")
def add_position(item: PortfolioItem):
    demo_portfolio.append(item.dict())
    return {"msg": "Added"}

@app.delete("/api/portfolio/{symbol}")
def delete_position(symbol: str):
    global demo_portfolio
    demo_portfolio = [p for p in demo_portfolio if p['symbol'] != symbol]
    return {"msg": "Deleted"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
