import uvicorn
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import datetime
from typing import List, Optional
from cachetools import TTLCache
from jose import JWTError, jwt
from passlib.context import CryptContext
import uuid

app = FastAPI(title="AI Stock Analysis API - SRS Ultimate")

# --- CORS 設定 (允許前端連線) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 模擬資料庫 (In-Memory) ---
# 注意：這是為了 Demo 方便，重啟後資料會清空
users_db = {} 
portfolio_db = {} # {username: [positions]}
favorites_db = {} # {username: [tickers]}

# --- Auth 設定 (JWT) ---
SECRET_KEY = "srs-super-secret-key-demo"
ALGORITHM = "HS256"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# --- 快取設定 ---
cache = TTLCache(maxsize=200, ttl=1800) # 個股分析快取 30 分鐘
rank_cache = TTLCache(maxsize=1, ttl=3600) # 排行榜快取 1 小時

# --- 資料模型 ---
class User(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class PortfolioItem(BaseModel):
    symbol: str
    cost_price: float
    shares: int
    date: str

class AnalyzeRequest(BaseModel):
    symbol: str
    principal: int = 100000
    risk: str = "neutral"

# --- Auth Helper Functions ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="無效憑證")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="憑證過期")

# --- 1. 真實數據運算核心 (SRS Requirement) ---

def calculate_technicals(df):
    """計算 RSI, MACD, BB, KD, MA"""
    if len(df) < 60: return None
    close = df['Close']
    high = df['High']
    low = df['Low']

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
    signal = macd.ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + (std20 * 2)
    lower = sma20 - (std20 * 2)

    # KD (簡易)
    low_min = low.rolling(9).min()
    high_max = high.rolling(9).max()
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
    """爬取證交所真實籌碼 (針對台股)"""
    if not (stock_code.isdigit() and len(stock_code) == 4): return 0
    try:
        url = "https://www.twse.com.tw/rwd/zh/fund/T86?response=json&selectType=ALL"
        # 設定 User-Agent 避免被擋
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=4)
        if res.status_code == 200:
            data = res.json()
            if 'data' in data:
                for row in data['data']:
                    if row[0] == stock_code:
                        # 回傳三大法人合計買賣超股數
                        return int(row[-1].replace(',', ''))
    except:
        pass
    return 0

def get_real_news(stock):
    """抓取 Yahoo 真實新聞"""
    try:
        news = stock.news
        result = []
        for n in news[:5]:
            result.append({
                "title": n.get('title', '無標題'),
                "link": n.get('link', '#'),
                "publisher": n.get('publisher', 'Yahoo')
            })
        return result
    except:
        return []

def calculate_var(returns, confidence_level=0.05):
    """計算 95% VaR (SRS #8 - 極端風險預警)"""
    if len(returns) < 1: return 0
    return returns.quantile(confidence_level)

def detect_patterns(df):
    """K線型態辨識 (SRS #15)"""
    patterns = []
    if len(df) < 5: return patterns
    last = df.iloc[-1]
    body = abs(last['Close'] - last['Open'])
    upper = last['High'] - max(last['Close'], last['Open'])
    lower = min(last['Close'], last['Open']) - last['Low']
    
    # 錘子線
    if lower > body * 2 and upper < body * 0.5: patterns.append("錘子線")
    # 長紅 K
    if last['Close'] > last['Open'] and body > (last['High'] - last['Low']) * 0.8: patterns.append("長紅 K")
    # 均線黃金交叉
    ma5 = df['Close'].rolling(5).mean()
    ma20 = df['Close'].rolling(20).mean()
    if ma5.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-2] <= ma20.iloc[-2]: patterns.append("均線黃金交叉")
    
    return patterns

# --- API Endpoints ---

@app.get("/")
def home():
    return {"status": "SRS_Backend_Ready"}

# 1. 會員系統 (SRS #13)
@app.post("/auth/register")
def register(user: User):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="User exists")
    users_db[user.username] = get_password_hash(user.password)
    portfolio_db[user.username] = []
    favorites_db[user.username] = []
    return {"message": "Success"}

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    pwd = users_db.get(form_data.username)
    if not pwd or not verify_password(form_data.password, pwd):
        raise HTTPException(status_code=401, detail="Auth Failed")
    token = create_access_token(data={"sub": form_data.username})
    return {"access_token": token, "token_type": "bearer"}

# 2. 深度分析 (SRS 核心 - 需登入)
@app.post("/api/analyze")
def analyze(req: AnalyzeRequest, user: str = Depends(get_current_user)):
    ticker = req.symbol.upper()
    
    # 在這裡不使用 Cache，確保 User 每次都能看到即時運算
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y") # 抓一年以計算長天期均線
        if hist.empty: raise HTTPException(status_code=404, detail="No Data")

        # --- A. 運算各項指標 ---
        tech_data = calculate_technicals(hist)
        
        # 基本面 (Yahoo Info)
        info = stock.info
        fund_score = 50
        if info.get('trailingEps', 0) > 0: fund_score += 20
        if info.get('returnOnEquity', 0) > 0.1: fund_score += 20
        pe = info.get('trailingPE', 0)
        if 0 < pe < 25: fund_score += 10
        fund_score = min(99, max(1, fund_score))

        # 籌碼面 (真實爬蟲)
        chip_val = 0
        if ".TW" in ticker:
            chip_val = get_twse_chip(ticker.split('.')[0])
        chip_score = 50
        if chip_val > 1000000: chip_score = 85
        elif chip_val > 0: chip_score = 60
        elif chip_val < -1000000: chip_score = 25
        elif chip_val < 0: chip_score = 40

        # 技術面評分
        tech_score = 50
        if tech_data:
            if tech_data['rsi'] > 70: tech_score = 85
            elif tech_data['rsi'] < 30: tech_score = 25
            if tech_data['price'] > tech_data['ma20']: tech_score += 10
            if tech_data['ma5'] > tech_data['ma20']: tech_score += 10
        tech_score = min(99, max(1, int(tech_score)))

        # 消息面 (真實新聞列表)
        news_list = get_real_news(stock)
        news_score = 50 # 中性

        # 總分
        total_score = int(tech_score*0.4 + fund_score*0.2 + chip_score*0.2 + news_score*0.2)

        # SRS #8 風險運算
        daily_returns = hist['Close'].pct_change().dropna()
        var_95 = calculate_var(daily_returns)
        max_loss = req.principal * abs(var_95)

        # SRS #6 建議價位 (ATR)
        high_low = hist['High'] - hist['Low']
        atr = high_low.rolling(14).mean().iloc[-1]
        price = hist['Close'].iloc[-1]

        # SRS #4 ROI 預估
        roi = {
            "1d": round(daily_returns.iloc[-1] * 100, 2),
            "1w": round(hist['Close'].pct_change(5).iloc[-1] * 100, 2),
            "1m": round(hist['Close'].pct_change(20).iloc[-1] * 100, 2),
            "1y": round(hist['Close'].pct_change(250).iloc[-1] * 100, 2)
        }

        # 圖表數據
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()

        return {
            "symbol": ticker,
            "current_price": round(price, 2),
            "ai_score": total_score,
            "evaluation": "多頭" if total_score > 60 else "空頭",
            "scores": {"tech": tech_score, "fund": fund_score, "chip": chip_score, "news": news_score},
            "roi": roi,
            "risk": {
                "var_95_pct": round(var_95 * 100, 2),
                "max_loss_est": round(max_loss, 0)
            },
            "strategy": {
                "entry": round(price, 2),
                "stop_loss": round(price - atr*2, 2),
                "take_profit": round(price + atr*3, 2)
            },
            "patterns": detect_patterns(hist),
            "tech_details": tech_data,
            "news": news_list,
            "chart_data": {
                "history_date": dates,
                "history_price": [round(p, 2) for p in prices]
            },
            # 兼容舊前端欄位
            "totalScore": total_score, 
            "currentPrice": round(price, 2),
            "recommendation": "買進" if total_score > 70 else "持有",
            "details": {"tech": tech_score, "fund": fund_score, "chip": chip_score, "news": news_score},
            "tech_indicators": tech_data,
            "news_list": news_list
        }

    except Exception as e:
        print(f"Analyze Error: {e}")
        raise HTTPException(status_code=500, detail="後端運算錯誤")

# 3. 排行榜 (SRS #11)
@app.get("/api/rankings")
def get_rankings(user: str = Depends(get_current_user)):
    if "global_rank" in rank_cache: return rank_cache["global_rank"]
    
    # 掃描熱門股
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
    
    # 依漲幅排序
    results.sort(key=lambda x: x['change_pct'], reverse=True)
    rank_cache["global_rank"] = results
    return results

# 4. User Data (收藏 & 資產)
@app.get("/api/favorites")
def get_favorites(user: str = Depends(get_current_user)):
    return favorites_db.get(user, [])

@app.post("/api/favorites/{symbol}")
def add_favorite(symbol: str, user: str = Depends(get_current_user)):
    favs = favorites_db.get(user, [])
    if symbol not in favs: favs.append(symbol)
    return favs

@app.delete("/api/favorites/{symbol}")
def remove_favorite(symbol: str, user: str = Depends(get_current_user)):
    favs = favorites_db.get(user, [])
    if symbol in favs: favs.remove(symbol)
    return favs

@app.get("/api/portfolio")
def get_portfolio(user: str = Depends(get_current_user)):
    return portfolio_db.get(user, [])

@app.post("/api/portfolio/add")
def add_position(item: PortfolioItem, user: str = Depends(get_current_user)):
    if user not in portfolio_db: portfolio_db[user] = []
    portfolio_db[user].append(item.dict())
    return {"msg": "Added"}

@app.delete("/api/portfolio/{symbol}")
def delete_position(symbol: str, user: str = Depends(get_current_user)):
    if user in portfolio_db:
        portfolio_db[user] = [p for p in portfolio_db[user] if p['symbol'] != symbol]
    return {"msg": "Deleted"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
