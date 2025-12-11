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

# ============================
# 基本設定與初始化
# ============================

app = FastAPI(title="AI Stock Analyzer Backend", version="1.0.0")

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


# ============================
# SQLite 資料庫初始化
# ============================

def init_db():
    conn = sqlite3.connect("stock_app.db")
    c = conn.cursor()
    # 使用者表
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users
        (username TEXT PRIMARY KEY, hashed_password TEXT)
    """
    )
    # 模擬持倉表
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         username TEXT,
         symbol TEXT,
         shares INTEGER,
         avg_cost REAL)
    """
    )
    # 收藏表
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS favorites
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         username TEXT,
         symbol TEXT)
    """
    )
    conn.commit()
    conn.close()


init_db()


# ============================
# Pydantic 模型定義
# ============================

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
    duration: str  # 目前前端仍使用字串，例如：'當沖', '短期', '中期', '長期'


class PortfolioItem(BaseModel):
    symbol: str
    shares: int
    cost: float


# ============================
# 共用輔助函式
# ============================

def get_db():
    conn = sqlite3.connect("stock_app.db")
    conn.row_factory = sqlite3.Row
    return conn


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
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


# ============================
# AI 分析與風險 / ROI 計算邏輯
# ============================

def compute_horizon_roi(prices: pd.Series, horizon: int) -> float:
    """
    以歷史資料估計「持有 horizon 天」的平均報酬率（百分比）。
    """
    if len(prices) <= horizon:
        return 0.0

    future_prices = prices.shift(-horizon)
    returns = (future_prices - prices) / prices
    returns = returns.dropna()
    if returns.empty:
        return 0.0

    return float(returns.mean()) * 100.0  # 轉為百分比


def calculate_ai_metrics(df: pd.DataFrame, symbol: str):
    """
    核心 AI 分析邏輯：
    - 技術指標 (RSI、MA)
    - 技術面評分
    - 四大面向分數（目前基本面 / 籌碼 / 消息面以固定邏輯模擬，可之後接真實資料）
    - ROI 估計（1 日 / 5 日 / 60 日 / 1 年）
    - VaR 95% 風險
    - 未來價格預測（線性回歸）
    """
    if len(df) < 60:
        return None

    df = df.sort_index()
    close_prices = df["Close"]
    returns = close_prices.pct_change().dropna()

    # -------- 技術指標：RSI --------
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    # -------- 均線 --------
    ma5 = float(close_prices.rolling(window=5).mean().iloc[-1])
    ma20 = float(close_prices.rolling(window=20).mean().iloc[-1])
    ma60 = float(close_prices.rolling(window=60).mean().iloc[-1])

    # -------- 技術面評分 (0~100) --------
    score = 50  # 基礎分

    # RSI 區間
    if current_rsi < 30:
        score += 10  # 超賣，反彈機率較高
    elif current_rsi > 70:
        score -= 10  # 超買，回檔風險較高

    # 均線多頭排列
    if ma5 > ma20:
        score += 10
    if ma20 > ma60:
        score += 10

    # 60 日趨勢
    if len(close_prices) >= 60:
        ret_60 = (close_prices.iloc[-1] - close_prices.iloc[-60]) / close_prices.iloc[-60]
        score += float(ret_60 * 50)  # 趨勢強度加減分

    # 波動度
    vol = returns.std() * np.sqrt(252)
    if vol < 0.2:
        score += 5  # 波動較小，穩定加分
    elif vol > 0.4:
        score -= 5  # 波動較大，風險加大

    technical_score = int(np.clip(score, 0, 100))

    # -------- 其它三面向目前先用簡化邏輯，可之後接真實財報 / 籌碼 / 新聞 --------
    fundamental_score = 70  # 先給中上水準，之後可接財報
    chip_score = 65  # 可改為依法人買賣超計算
    news_score = 60  # 可接新聞情緒分析

    # 綜合 AI 分數（四面向平均）
    ai_score = int(
        np.clip(
            (technical_score + fundamental_score + chip_score + news_score) / 4.0, 0, 100
        )
    )

    # 評語
    if ai_score >= 80:
        sentiment = "強力看多"
    elif ai_score >= 60:
        sentiment = "偏多"
    elif ai_score >= 40:
        sentiment = "中立"
    else:
        sentiment = "偏空"

    # -------- ROI 估計：使用歷史分佈估算平均報酬 --------
    roi_1d = compute_horizon_roi(close_prices, 1)
    roi_5d = compute_horizon_roi(close_prices, 5)
    roi_60d = compute_horizon_roi(close_prices, 60)
    roi_1y = compute_horizon_roi(close_prices, 252)  # 約一年交易日

    roi_dict = {
        "day_1": round(roi_1d, 2),
        "day_5": round(roi_5d, 2),
        "day_60": round(roi_60d, 2),
        "day_365": round(roi_1y, 2),
    }

    # -------- VaR 95% 風險估計（單日） --------
    if returns.empty:
        var_95 = 0.0
    else:
        sorted_returns = returns.sort_values()
        var_95 = float(sorted_returns.quantile(0.05))  # 5% 分位數

    var_95_percent = var_95 * 100.0
    current_price = float(close_prices.iloc[-1])
    pessimistic_price = current_price * (1 + var_95)

    # -------- 未來價格預測（線性回歸，for K 線預測區間） --------
    window = min(120, len(close_prices))
    X = np.arange(window).reshape(-1, 1)
    y = close_prices.iloc[-window:].values
    model = LinearRegression()
    model.fit(X, y)

    future_days = 60  # 預測 60 天
    future_X = np.arange(window, window + future_days).reshape(-1, 1)
    future_pred = model.predict(future_X)

    last_date = df.index[-1]
    prediction_data = []
    for i, price in enumerate(future_pred):
        next_date = last_date + datetime.timedelta(days=i + 1)
        prediction_data.append(
            {
                "date": next_date.strftime("%Y-%m-%d"),
                "predicted_price": round(float(price), 2),
            }
        )

    # -------- 歷史 K 線資料（給圖用，先用收盤價） --------
    history_window = min(300, len(close_prices))
    history_data = [
        {"date": d.strftime("%Y-%m-%d"), "price": round(float(p), 2)}
        for d, p in zip(df.index[-history_window:], close_prices[-history_window:])
    ]

    return {
        "current_price": round(current_price, 2),
        "ai_score": ai_score,
        "sentiment": sentiment,
        "technical_score": technical_score,
        "fundamental_score": fundamental_score,
        "chip_score": chip_score,
        "news_score": news_score,
        "rsi": round(current_rsi, 2),
        "ma5": round(ma5, 2),
        "ma20": round(ma20, 2),
        "ma60": round(ma60, 2),
        "roi": roi_dict,
        "var_95_percent": round(var_95_percent, 2),
        "pessimistic_price": round(pessimistic_price, 2),
        "history": history_data,
        "prediction": prediction_data,
    }


# ============================
# API Endpoints
# ============================

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (form_data.username,))
    user_row = c.fetchone()
    conn.close()

    if not user_row or not verify_password(form_data.password, user_row["hashed_password"]):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": user_row["username"]})
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/register")
async def register(user: User):
    conn = get_db()
    c = conn.cursor()
    try:
        hashed_pw = get_password_hash(user.password)
        c.execute(
            "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
            (user.username, hashed_pw),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=400, detail="Username already exists")
    conn.close()
    return {"message": "User created successfully"}


@app.post("/api/analyze")
async def analyze_stock(request: AnalysisRequest):
    try:
        # 1. 取得較長區間的歷史資料，用於 ROI / VaR / 預測
        ticker = yf.Ticker(request.symbol)
        df = ticker.history(period="5y")  # 比 1 年更長，模型更穩定

        if df.empty:
            raise HTTPException(status_code=404, detail="Stock not found")

        # 2. AI 核心分析
        ai_result = calculate_ai_metrics(df, request.symbol)
        if not ai_result:
            raise HTTPException(
                status_code=400, detail="Not enough data for analysis"
            )

        current_price = ai_result["current_price"]

        # 3. 資金配置試算
        principal = request.principal
        max_shares = int(principal // current_price) if current_price > 0 else 0
        cost = max_shares * current_price

        # 4. 波段建議價位
        buy_price = current_price
        take_profit = buy_price * 1.20
        stop_loss = buy_price * 0.90

        # 5. ROI 預估金額（根據歷史平均報酬率）
        roi_pct = ai_result["roi"]  # 百分比
        roi_day_amt = cost * roi_pct["day_1"] / 100.0
        roi_week_amt = cost * roi_pct["day_5"] / 100.0
        roi_60_amt = cost * roi_pct["day_60"] / 100.0
        roi_year_amt = cost * roi_pct["day_365"] / 100.0

        # 6. 極端行情風險估計（以 VaR 95% 百分比估計）
        max_drawdown_pct = ai_result["var_95_percent"]
        max_loss_amt = cost * abs(max_drawdown_pct) / 100.0

        return {
            "symbol": request.symbol.upper(),
            "price": current_price,
            "ai_score": ai_result["ai_score"],
            "ai_sentiment": ai_result["sentiment"],
            # 四大面向分數
            "score_breakdown": {
                "technical": ai_result["technical_score"],
                "fundamental": ai_result["fundamental_score"],
                "chip": ai_result["chip_score"],
                "news": ai_result["news_score"],
            },
            # 常用技術指標
            "technical": {
                "rsi": ai_result["rsi"],
                "ma5": ai_result["ma5"],
                "ma20": ai_result["ma20"],
                "ma60": ai_result["ma60"],
            },
            # 資金配置試算
            "money_management": {
                "principal": principal,
                "max_shares": max_shares,
                "total_cost": round(cost, 2),
                "risk_loss_10_percent": round(cost * 0.1, 2),
            },
            # 波段操作建議
            "advice": {
                "buy_price": round(buy_price, 2),
                "take_profit": round(take_profit, 2),
                "stop_loss": round(stop_loss, 2),
            },
            # ROI 模組（百分比 + 金額）
            "roi_estimates": {
                # 保留原本 key 名稱 day/week/month/year，方便前端沿用
                "day": {"amt": round(roi_day_amt, 2), "pct": roi_pct["day_1"]},
                "week": {"amt": round(roi_week_amt, 2), "pct": roi_pct["day_5"]},
                "month": {"amt": round(roi_60_amt, 2), "pct": roi_pct["day_60"]},
                "year": {"amt": round(roi_year_amt, 2), "pct": roi_pct["day_365"]},
            },
            # 極端行情預警（風險模型）
            "risk_analysis": {
                "max_drawdown_pct": max_drawdown_pct,
                "max_loss_amt": round(max_loss_amt, 2),
                "pessimistic_price": ai_result["pessimistic_price"],
            },
            # 圖表資料：歷史 + 預測
            "chart_data": {
                "history": ai_result["history"],
                "prediction": ai_result["prediction"],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news")
async def get_news():
    # 模擬新聞數據
    # 若未來串接真實新聞，可改為呼叫外部 API 或 yfinance 的 news 屬性
    return [
        {"time": "剛剛", "title": "半導體庫存去化順利，下半年展望樂觀", "source": "產業快訊"},
        {"time": "10 分鐘前", "title": "電動車市場競爭白熱化，車廠降價搶市佔", "source": "產業快訊"},
        {"time": "30 分鐘前", "title": "中東地緣政治緊張，油價波動加劇", "source": "國際要聞"},
    ]


# ============================
# 模擬資產相關 API（需登入）
# ============================

@app.get("/api/portfolio")
async def get_portfolio(user: str = Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM portfolio WHERE username=?", (user,))
    rows = c.fetchall()

    portfolio = []
    total_asset = 0.0
    total_cost = 0.0

    for row in rows:
        # 簡化版本：不即時抓現價，以成本 * 1.05 模擬
        current_price = row["avg_cost"] * 1.05
        value = row["shares"] * current_price
        cost = row["shares"] * row["avg_cost"]
        portfolio.append(
            {
                "symbol": row["symbol"],
                "shares": row["shares"],
                "cost": row["avg_cost"],
                "market_value": round(value, 2),
                "pnl": round(value - cost, 2),
            }
        )
        total_asset += value
        total_cost += cost

    conn.close()
    return {
        "total_asset": round(total_asset, 2),
        "total_cost": round(total_cost, 2),
        "unrealized_pnl": round(total_asset - total_cost, 2),
        "holdings": portfolio,
    }


@app.post("/api/portfolio/add")
async def add_to_portfolio(item: PortfolioItem, user: str = Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO portfolio (username, symbol, shares, avg_cost) VALUES (?, ?, ?, ?)",
        (user, item.symbol, item.shares, item.cost),
    )
    conn.commit()
    conn.close()
    return {"message": "Added to portfolio"}


# ============================
# （可選）主程式啟動
# ============================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

