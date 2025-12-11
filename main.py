import sqlite3
import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# --- 設定與初始化 ---
app = FastAPI()

# 允許 CORS (讓前端 Vercel 可以呼叫)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境建議改為前端的 Vercel 網址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT 設定
SECRET_KEY = "your_secret_key_here_please_change"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 資料庫初始化
def init_db():
    conn = sqlite3.connect('stock_app.db')
    c = conn.cursor()
    # 使用者表
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, hashed_password TEXT)''')
    # 模擬持倉表
    c.execute('''CREATE TABLE IF NOT EXISTS portfolio
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, 
                  symbol TEXT, shares INTEGER, avg_cost REAL)''')
    # 收藏表
    c.execute('''CREATE TABLE IF NOT EXISTS favorites
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, symbol TEXT)''')
    conn.commit()
    conn.close()

init_db()

# --- 模型定義 (Pydantic) ---
class User(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class AnalysisRequest(BaseModel):
    symbol: str
    principal: float
    strategy: str
    duration: str

class PortfolioItem(BaseModel):
    symbol: str
    shares: int
    cost: float

# --- 輔助函式 ---
def get_db():
    conn = sqlite3.connect('stock_app.db')
    conn.row_factory = sqlite3.Row
    return conn

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
        detail="Could not validate credentials",
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

# --- 核心 AI 分析邏輯 ---
def calculate_ai_metrics(df: pd.DataFrame, symbol: str):
    """
    計算 AI 評分、預測與風險指標
    """
    if len(df) < 60:
        return None

    # 1. 準備數據
    close_prices = df['Close'].values
    
    # 2. 技術指標計算 (簡化版)
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    # MA (均線)
    ma5 = df['Close'].rolling(window=5).mean().iloc[-1]
    ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
    ma60 = df['Close'].rolling(window=60).mean().iloc[-1]

    # 3. AI 評分 (0-100)
    score = 50  # 基礎分
    # 技術面權重
    if current_rsi < 30: score += 10 # 超賣反彈
    elif current_rsi > 70: score -= 5 # 超買風險
    if ma5 > ma20: score += 10 # 短期多頭
    if ma20 > ma60: score += 10 # 中期多頭
    
    # 簡單動能
    returns = df['Close'].pct_change()
    volatility = returns.std() * np.sqrt(252)
    if volatility < 0.2: score += 5 # 波動穩定加分

    score = min(max(int(score), 0), 100)
    
    # 評語
    if score >= 80: sentiment = "強力看多"
    elif score >= 60: sentiment = "偏多"
    elif score >= 40: sentiment = "中立"
    else: sentiment = "偏空"

    # 4. 股價預測 (線性回歸模擬)
    # 使用最近 30 天預測未來 10 天
    X = np.arange(len(df))[-30:].reshape(-1, 1)
    y = close_prices[-30:]
    model = LinearRegression()
    model.fit(X, y)
    
    future_days = 10
    future_X = np.arange(len(df), len(df) + future_days).reshape(-1, 1)
    future_pred = model.predict(future_X)
    
    prediction_data = []
    last_date = df.index[-1]
    for i, price in enumerate(future_pred):
        next_date = last_date + datetime.timedelta(days=i+1)
        prediction_data.append({
            "date": next_date.strftime('%Y-%m-%d'),
            "predicted_price": round(price, 2)
        })

    # 5. 極端行情風險 (VaR 95%)
    # 歷史模擬法
    sorted_returns = returns.dropna().sort_values()
    var_95 = sorted_returns.quantile(0.05) # 95% 信心水準的單日最大跌幅
    
    current_price = close_prices[-1]
    pessimistic_price = current_price * (1 + var_95)

    return {
        "current_price": round(current_price, 2),
        "score": score,
        "sentiment": sentiment,
        "rsi": round(current_rsi, 2),
        "ma5": round(ma5, 2),
        "ma20": round(ma20, 2),
        "var_95_percent": round(var_95 * 100, 2),
        "pessimistic_price": round(pessimistic_price, 2),
        "prediction": prediction_data,
        "history": [{"date": d.strftime('%Y-%m-%d'), "price": round(p, 2)} for d, p in zip(df.index[-60:], df['Close'][-60:])]
    }

# --- API Endpoints ---

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (form_data.username,))
    user_row = c.fetchone()
    conn.close()
    
    if not user_row or not verify_password(form_data.password, user_row['hashed_password']):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user_row['username']})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register(user: User):
    conn = get_db()
    c = conn.cursor()
    try:
        hashed_pw = get_password_hash(user.password)
        c.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)", 
                  (user.username, hashed_pw))
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")
    conn.close()
    return {"message": "User created successfully"}

@app.post("/api/analyze")
async def analyze_stock(request: AnalysisRequest):
    try:
        # 1. 獲取數據
        ticker = yf.Ticker(request.symbol)
        df = ticker.history(period="1y")
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Stock not found")

        # 2. 執行 AI 分析
        ai_result = calculate_ai_metrics(df, request.symbol)
        if not ai_result:
            raise HTTPException(status_code=400, detail="Not enough data for analysis")

        current_price = ai_result['current_price']
        
        # 3. 計算資金配置
        max_shares = int(request.principal // current_price)
        cost = max_shares * current_price
        
        # 4. 波段建議
        buy_price = current_price
        take_profit = buy_price * 1.20
        stop_loss = buy_price * 0.90
        
        # 5. ROI 預估 (ROI 模組)
        # 這裡做簡化假設：假設每日平均波動為 1.5%
        roi_day = cost * 0.015
        roi_week = cost * 0.04
        roi_month = cost * 0.12
        roi_year = cost * 0.25 # 假設年化 25%

        return {
            "symbol": request.symbol.upper(),
            "price": current_price,
            "ai_score": ai_result['score'],
            "ai_sentiment": ai_result['sentiment'],
            "technical": {
                "rsi": ai_result['rsi'],
                "ma5": ai_result['ma5'],
                "ma20": ai_result['ma20']
            },
            "money_management": {
                "principal": request.principal,
                "max_shares": max_shares,
                "total_cost": cost,
                "risk_loss_10_percent": cost * 0.1
            },
            "advice": {
                "buy_price": buy_price,
                "take_profit": round(take_profit, 2),
                "stop_loss": round(stop_loss, 2)
            },
            "roi_estimates": {
                "day": {"amt": round(roi_day), "pct": 1.5},
                "week": {"amt": round(roi_week), "pct": 4.0},
                "month": {"amt": round(roi_month), "pct": 12.0},
                "year": {"amt": round(roi_year), "pct": 25.0}
            },
            "risk_analysis": {
                "max_drawdown_pct": ai_result['var_95_percent'],
                "max_loss_amt": round(cost * (abs(ai_result['var_95_percent'])/100)),
                "pessimistic_price": ai_result['pessimistic_price']
            },
            "chart_data": {
                "history": ai_result['history'],
                "prediction": ai_result['prediction']
            }
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/news")
async def get_news():
    # 模擬新聞數據
    # 在真實環境中，這裡可以接 Google News API 或 yfinance 的 news 屬性
    return [
        {"time": "3 分鐘前", "title": "聯準會暗示暫停升息，科技股大漲", "source": "Bloomberg"},
        {"time": "15 分鐘前", "title": "AI 晶片需求強勁，供應鏈產能滿載", "source": "Reuters"},
        {"time": "1 小時前", "title": "地緣政治風險升溫，油價波動加劇", "source": "CNBC"},
    ]

# 模擬資產相關 API
@app.get("/api/portfolio")
async def get_portfolio(user: str = Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM portfolio WHERE username=?", (user,))
    rows = c.fetchall()
    portfolio = []
    total_asset = 0
    total_cost = 0
    
    for row in rows:
        # 簡單起見，這裡不即時抓現價，實際應再 call yfinance
        current_price = row['avg_cost'] * 1.05 # 假裝賺 5%
        value = row['shares'] * current_price
        portfolio.append({
            "symbol": row['symbol'],
            "shares": row['shares'],
            "cost": row['avg_cost'],
            "market_value": round(value),
            "pnl": round(value - (row['shares'] * row['avg_cost']))
        })
        total_asset += value
        total_cost += (row['shares'] * row['avg_cost'])
        
    conn.close()
    return {
        "total_asset": round(total_asset),
        "total_cost": round(total_cost),
        "unrealized_pnl": round(total_asset - total_cost),
        "holdings": portfolio
    }

@app.post("/api/portfolio/add")
async def add_to_portfolio(item: PortfolioItem, user: str = Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO portfolio (username, symbol, shares, avg_cost) VALUES (?, ?, ?, ?)",
              (user, item.symbol, item.shares, item.cost))
    conn.commit()
    conn.close()
    return {"message": "Added to portfolio"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
