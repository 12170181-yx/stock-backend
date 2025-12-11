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
from typing import List, Optional, Dict
from cachetools import TTLCache
from jose import JWTError, jwt
from passlib.context import CryptContext
import uuid

# --- 初始化 APP ---
app = FastAPI(title="AI Stock Analysis API - SRS Ultimate")

# --- CORS 設定 (允許前端連線) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 模擬資料庫 (In-Memory DB) ---
# 注意：Render 免費版重啟後資料會清空，僅供 Demo 與開發測試
users_db = {}  # {username: hashed_password}
portfolio_db = {} # {username: [positions]}
favorites_db = {} # {username: [tickers]}

# --- Auth 設定 (JWT) ---
SECRET_KEY = "srs-secret-key-demo-change-me"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 # Token 有效期 60 分鐘
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

# --- 快取設定 ---
cache = TTLCache(maxsize=200, ttl=1800) # 個股分析快取 30 分鐘
rank_cache = TTLCache(maxsize=1, ttl=3600) # 排行榜快取 1 小時

# --- 資料模型 (Pydantic Models) ---
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

# --- Auth 輔助函式 ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="無法驗證憑證",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username

# --- 核心演算法 (SRS Algorithms) ---

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

def calculate_var(returns, confidence_level=0.05):
    """SRS #8: 計算歷史模擬法 VaR (95% 信心水準)"""
    if len(returns) < 1: return 0
    # 取分位數，例如 5% 最差的報酬率
    return returns.quantile(confidence_level)

def detect_patterns(df):
    """SRS #15: K 線型態辨識"""
    patterns = []
    if len(df) < 5: return patterns
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 計算實體與影線
    body = abs(last['Close'] - last['Open'])
    upper_shadow = last['High'] - max(last['Close'], last['Open'])
    lower_shadow = min(last['Close'], last['Open']) - last['Low']
    avg_body = abs(df['Close'] - df['Open']).mean()

    # 1. 錘子線 (Hammer) - 底部反轉訊號
    if lower_shadow > body * 2 and upper_shadow < body * 0.5:
        patterns.append("錘子線 (反轉)")
        
    # 2. 長紅 K (Bullish Marubozu) - 強勢多頭
    if last['Close'] > last['Open'] and body > avg_body * 1.5 and body > (last['High'] - last['Low']) * 0.8:
        patterns.append("長紅 K (強勢)")

    # 3. 均線黃金交叉
    ma5 = df['Close'].rolling(5).mean()
    ma20 = df['Close'].rolling(20).mean()
    if ma5.iloc[-1] > ma20.iloc[-1] and ma5.iloc[-2] <= ma20.iloc[-2]:
        patterns.append("MA黃金交叉")

    return patterns

def get_twse_chip(stock_code):
    """爬取證交所真實籌碼 (針對台股)"""
    if not (stock_code.isdigit() and len(stock_code) == 4): return 0
    try:
        url = "https://www.twse.com.tw/rwd/zh/fund/T86?response=json&selectType=ALL"
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
    """SRS #2: 真實新聞抓取"""
    try:
        news = stock.news
        result = []
        for n in news[:5]:
            result.append({
                "title": n.get('title'),
                "link": n.get('link'),
                "publisher": n.get('publisher'),
                "time": "Latest"
            })
        return result
    except:
        return []

# --- API Endpoints ---

@app.get("/")
def home():
    return {"status": "SRS_Backend_Ready", "version": "Ultimate_v1"}

# 1. 認證系統 (SRS #13)
@app.post("/auth/register")
def register(user: User):
    if user.username in users_db:
        raise HTTPException(status_code=400, detail="用戶已存在")
    users_db[user.username] = get_password_hash(user.password)
    # 初始化使用者資料空間
    portfolio_db[user.username] = []
    favorites_db[user.username] = []
    return {"message": "註冊成功"}

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user_password = users_db.get(form_data.username)
    if not user_password or not verify_password(form_data.password, user_password):
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/auth/me")
def read_users_me(current_user: str = Depends(get_current_user)):
    return {"username": current_user}

# 2. 深度分析 API (SRS 核心功能)
@app.post("/api/analyze")
def analyze_stock(req: AnalyzeRequest, user: str = Depends(get_current_user)):
    ticker = req.symbol.upper()
    
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty: raise HTTPException(status_code=404, detail="查無資料")

        # --- A. 基礎數據運算 ---
        close = hist['Close']
        tech_data = calculate_technicals(hist)

        # --- B. SRS #8 極端行情預警 (VaR) ---
        daily_returns = close.pct_change().dropna()
        var_95 = calculate_var(daily_returns, 0.05)
        max_loss_est = req.principal * abs(var_95)

        # --- C. SRS #4 ROI 預估 ---
        roi_data = {
            "1d": round(daily_returns.iloc[-1] * 100, 2),
            "1w": round(close.pct_change(5).iloc[-1] * 100, 2),
            "1m": round(close.pct_change(20).iloc[-1] * 100, 2),
            "1y": round(close.pct_change(250).iloc[-1] * 100, 2)
        }

        # --- D. SRS #6 波段建議價位 (ATR 策略) ---
        high_low = hist['High'] - hist['Low']
        atr = high_low.rolling(14).mean().iloc[-1]
        current_price = close.iloc[-1]
        
        entry = current_price
        stop_loss = current_price - (atr * 2) # 2倍 ATR 停損
        take_profit = current_price + (atr * 3) # 3倍 ATR 停利

        # --- E. SRS #3 四大面向評分 ---
        # 技術面
        tech_score = 50
        if tech_data:
            if tech_data['rsi'] > 70: tech_score = 85
            elif tech_data['rsi'] < 30: tech_score = 25
            else: tech_score = 50 + (tech_data['rsi'] - 50) * 0.5
            if tech_data['price'] > tech_data['ma20']: tech_score += 10
        tech_score = min(99, max(1, int(tech_score)))

        # 基本面
        info = stock.info
        fund_score = 50
        if info.get('trailingEps', 0) > 0: fund_score += 20
        if info.get('returnOnEquity', 0) > 0.1: fund_score += 20
        if 0 < info.get('trailingPE', 0) < 25: fund_score += 10
        fund_score = min(99, max(1, fund_score))

        # 籌碼面 (台股真實)
        chip_val = 0
        if ".TW" in ticker:
            chip_val = get_twse_chip(ticker.split('.')[0])
        chip_score = 50
        if chip_val > 1000000: chip_score = 85
        elif chip_val > 0: chip_score = 60
        elif chip_val < -1000000: chip_score = 25
        elif chip_val < 0: chip_score = 40

        # 消息面
        news_list = get_real_news(stock)
        news_score = 50 # 這裡保持中性

        total_score = int(tech_score*0.4 + fund_score*0.2 + chip_score*0.2 + news_score*0.2)

        # --- F. SRS #15 K線型態 ---
        patterns = detect_patterns(hist)

        # --- G. 準備圖表數據 ---
        dates = hist.index.strftime('%Y-%m-%d').tolist()
        prices = hist['Close'].tolist()
        
        # 未來預測線
        last_date = datetime.datetime.strptime(dates[-1], '%Y-%m-%d')
        trend = 1.002 if tech_score > 60 else 0.998
        future_dates = []
        future_means = []
        for i in range(1, 6):
            future_dates.append((last_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d'))
            future_means.append(round(current_price * (trend ** i), 2))

        # --- H. 回傳完整 JSON ---
        return {
            "symbol": ticker,
            "current_price": round(current_price, 2),
            "ai_score": total_score,
            "evaluation": "多頭" if total_score > 60 else "空頭",
            "recommendation": "買進" if total_score > 70 else "持有",
            "scores": {"tech": tech_score, "fund": fund_score, "chip": chip_score, "news": news_score},
            "roi": roi_data,
            "risk": {
                "var_95_pct": round(var_95 * 100, 2),
                "max_loss_est": round(max_loss_est, 0)
            },
            "strategy": {
                "entry": round(entry, 2),
                "stop_loss": round(stop_loss, 2),
                "take_profit": round(take_profit, 2)
            },
            "patterns": patterns,
            "tech_details": tech_data,
            "news_list": news_list,
            "chart_data": {
                "history_date": dates,
                "history_price": [round(p, 2) for p in prices],
                "future_date": future_dates,
                "future_mean": future_means
            }
        }

    except Exception as e:
        print(f"Analyze Error: {e}")
        raise HTTPException(status_code=500, detail="分析失敗")

# 3. 排行榜 (SRS #2 市場快訊)
@app.get("/api/rankings")
def get_rankings(user: str = Depends(get_current_user)):
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

# 4. User Data (SRS #12, #14) - 需登入
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
