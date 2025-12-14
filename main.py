import sqlite3
import datetime
from typing import List
from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# =========================================================
# 基本設定
# =========================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = "PLEASE_CHANGE_THIS_SECRET_KEY"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# =========================================================
# Database
# =========================================================
def init_db():
    conn = sqlite3.connect("stock_app.db")
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            hashed_password TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            symbol TEXT
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            symbol TEXT,
            shares INTEGER,
            avg_cost REAL
        )
    """)

    conn.commit()
    conn.close()

init_db()

def get_db():
    conn = sqlite3.connect("stock_app.db")
    conn.row_factory = sqlite3.Row
    return conn

# =========================================================
# Models
# =========================================================
class User(BaseModel):
    username: str
    password: str

class AnalysisRequest(BaseModel):
    symbol: str
    principal: float
    strategy: str
    duration: str

class FavoriteReq(BaseModel):
    symbol: str

class PortfolioItem(BaseModel):
    symbol: str
    shares: int
    cost: float

# =========================================================
# Auth helpers
# =========================================================
def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401)
        return username
    except JWTError:
        raise HTTPException(status_code=401)

# =========================================================
# 技術指標
# =========================================================
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# =========================================================
# AI 核心分析（穩定、可重現）
# =========================================================
def analyze_stock_core(df: pd.DataFrame):
    close = df["Close"]

    rsi = calc_rsi(close).iloc[-1]
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1]

    returns = close.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252)

    # ---- 四大面向評分（固定規則）----
    technical = 50
    if rsi < 30: technical += 20
    if rsi > 70: technical -= 10
    if ma5 > ma20: technical += 10
    if ma20 > ma60: technical += 10
    technical = min(max(technical, 0), 100)

    fundamental = 60  # demo：未來可接財報 API
    chip = 70         # demo：台股籌碼可擴充
    news = 65         # demo：新聞情緒

    ai_score = round((technical + fundamental + chip + news) / 4)

    if ai_score >= 80:
        sentiment = "偏多（可積極）"
    elif ai_score >= 60:
        sentiment = "偏多"
    elif ai_score >= 40:
        sentiment = "中立"
    else:
        sentiment = "偏空"

    # ---- 預測（線性回歸，固定資料窗）----
    X = np.arange(len(close))[-30:].reshape(-1, 1)
    y = close[-30:].values
    model = LinearRegression().fit(X, y)

    future_X = np.arange(len(close), len(close) + 10).reshape(-1, 1)
    preds = model.predict(future_X)

    prediction = []
    last_date = df.index[-1]
    for i, p in enumerate(preds):
        prediction.append({
            "date": (last_date + datetime.timedelta(days=i+1)).strftime("%Y-%m-%d"),
            "predicted_price": round(float(p), 2)
        })

    # ---- VaR 95% ----
    var_95 = returns.quantile(0.05)

    return {
        "price": round(float(close.iloc[-1]), 2),
        "score": ai_score,
        "sentiment": sentiment,
        "score_breakdown": {
            "technical": technical,
            "fundamental": fundamental,
            "chip": chip,
            "news": news
        },
        "risk": {
            "var_pct": round(var_95 * 100, 2),
            "pessimistic_price": round(close.iloc[-1] * (1 + var_95), 2)
        },
        "history": [
            {"date": d.strftime("%Y-%m-%d"), "price": round(float(p), 2)}
            for d, p in zip(df.index[-60:], close[-60:])
        ],
        "prediction": prediction
    }

# =========================================================
# Auth API
# =========================================================
@app.post("/token")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (form.username,))
    user = c.fetchone()
    conn.close()

    if not user or not verify_password(form.password, user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": form.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/register")
async def register(user: User):
    conn = get_db()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users VALUES (?,?)",
            (user.username, get_password_hash(user.password))
        )
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(400, "Username exists")
    finally:
        conn.close()
    return {"message": "ok"}

# =========================================================
# Analysis API
# =========================================================
@app.post("/api/analyze")
async def analyze(req: AnalysisRequest):
    ticker = yf.Ticker(req.symbol)
    df = ticker.history(period="1y")

    if df.empty or len(df) < 60:
        raise HTTPException(400, "Not enough data")

    core = analyze_stock_core(df)
    price = core["price"]

    max_shares = int(req.principal // price)
    cost = max_shares * price

    return {
        "symbol": req.symbol.upper(),
        "price": price,
        "ai_score": core["score"],
        "ai_sentiment": core["sentiment"],
        "score_breakdown": core["score_breakdown"],
        "roi_estimates": {
            "day": {"pct": 1.5, "amt": round(cost * 0.015)},
            "week": {"pct": 4.0, "amt": round(cost * 0.04)},
            "month": {"pct": 12.0, "amt": round(cost * 0.12)},
            "year": {"pct": 25.0, "amt": round(cost * 0.25)}
        },
        "money_management": {
            "max_shares": max_shares,
            "total_cost": round(cost),
            "risk_loss_10_percent": round(cost * 0.1)
        },
        "advice": {
            "buy_price": price,
            "take_profit": round(price * 1.2, 2),
            "stop_loss": round(price * 0.9, 2)
        },
        "risk_analysis": {
            "max_drawdown_pct": core["risk"]["var_pct"],
            "max_loss_amt": round(cost * abs(core["risk"]["var_pct"]) / 100),
            "pessimistic_price": core["risk"]["pessimistic_price"]
        },
        "chart_data": {
            "history": core["history"],
            "prediction": core["prediction"]
        }
    }

# =========================================================
# News（真實）
# =========================================================
@app.get("/api/news")
async def news():
    t = yf.Ticker("AAPL")
    items = []
    for n in t.news[:10]:
        items.append({
            "title": n.get("title"),
            "time": "即時",
            "source": n.get("publisher"),
            "link": n.get("link")
        })
    return items

# =========================================================
# Favorites（需登入）
# =========================================================
@app.get("/api/favorites")
async def list_fav(user=Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT symbol FROM favorites WHERE username=?", (user,))
    rows = [r["symbol"] for r in c.fetchall()]
    conn.close()
    return {"favorites": rows}

@app.post("/api/favorites/add")
async def add_fav(req: FavoriteReq, user=Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute("INSERT INTO favorites(username,symbol) VALUES (?,?)", (user, req.symbol))
    conn.commit()
    conn.close()
    return {"ok": True}

@app.post("/api/favorites/remove")
async def remove_fav(req: FavoriteReq, user=Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM favorites WHERE username=? AND symbol=?", (user, req.symbol))
    conn.commit()
    conn.close()
    return {"ok": True}

# =========================================================
# Portfolio（需登入）
# =========================================================
@app.get("/api/portfolio")
async def portfolio(user=Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM portfolio WHERE username=?", (user,))
    rows = c.fetchall()
    conn.close()

    total_cost = sum(r["shares"] * r["avg_cost"] for r in rows)
    total_asset = round(total_cost * 1.05)

    return {
        "total_asset": total_asset,
        "total_cost": round(total_cost),
        "unrealized_pnl": round(total_asset - total_cost),
        "holdings": [
            {
                "symbol": r["symbol"],
                "shares": r["shares"],
                "cost": r["avg_cost"],
                "market_value": round(r["shares"] * r["avg_cost"] * 1.05),
                "pnl": round(r["shares"] * r["avg_cost"] * 0.05)
            } for r in rows
        ]
    }

# =========================================================
# K 線詳細分析（需登入）
# =========================================================
@app.get("/api/kline-detail")
async def kline_detail(
    symbol: str = Query(...),
    interval: str = Query("1d"),
    user=Depends(get_current_user)
):
    df = yf.Ticker(symbol).history(period="6mo", interval=interval)
    if df.empty:
        raise HTTPException(404, "No data")

    candles = []
    for d, r in df.tail(60).iterrows():
        candles.append({
            "date": d.strftime("%Y-%m-%d"),
            "open": round(float(r["Open"]), 2),
            "high": round(float(r["High"]), 2),
            "low": round(float(r["Low"]), 2),
            "close": round(float(r["Close"]), 2),
            "volume": int(r["Volume"])
        })

    # 型態偵測（可擴充到 48 種）
    patterns = []
    if candles[-1]["close"] > candles[-1]["open"]:
        patterns.append({"date": candles[-1]["date"], "pattern": "陽線（多方）"})

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "candles": candles,
        "patterns": patterns
    }
