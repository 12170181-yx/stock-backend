import sqlite3
import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import (
    FastAPI,
    HTTPException,
    Depends,
    status,
    Query,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression

# =====================================
# 基本設定與初始化
# =====================================

app = FastAPI(title="AI Stock Analyzer Backend", version="1.0.0")

# 允許 CORS (讓前端 Vercel 可以呼叫)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 正式上線可改成前端網址
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


# =====================================
# SQLite 資料庫初始化
# =====================================

def init_db():
    conn = sqlite3.connect("stock_app.db")
    c = conn.cursor()
    # 使用者表
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users
        (username TEXT PRIMARY KEY,
         hashed_password TEXT)
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


# =====================================
# Pydantic 模型
# =====================================

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
    duration: str  # 目前前端是字串：當沖 / 短期 / 中期 / 長期


class PortfolioItem(BaseModel):
    symbol: str
    shares: int
    cost: float


class FavoriteItem(BaseModel):
    symbol: str


# =====================================
# 共用輔助函式
# =====================================

def get_db():
    conn = sqlite3.connect("stock_app.db")
    conn.row_factory = sqlite3.Row
    return conn


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(
        minutes=ACCESS_TOKEN_EXPIRE_MINUTES
    )
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username


# =====================================
# AI 分析與技術指標計算
# =====================================

def compute_horizon_roi(prices: pd.Series, horizon: int) -> float:
    """
    用歷史資料估計「持有 horizon 天」的平均報酬率（百分比）。
    """
    if len(prices) <= horizon:
        return 0.0

    future_prices = prices.shift(-horizon)
    returns = (future_prices - prices) / prices
    returns = returns.dropna()
    if returns.empty:
        return 0.0

    return float(returns.mean()) * 100.0  # 轉百分比


def calculate_ai_metrics(df: pd.DataFrame, symbol: str):
    """
    核心 AI 分析邏輯：
    - 技術指標 (RSI、MA)
    - 四大面向評分
    - ROI 估計（1 日 / 5 日 / 60 日 / 1 年）
    - VaR 95% 風險
    - 未來價格預測（線性回歸）
    - 歷史走勢資料
    """
    if len(df) < 60:
        return None

    df = df.sort_index()
    close = df["Close"]
    returns = close.pct_change().dropna()

    # ---------- RSI ----------
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    # ---------- 均線 ----------
    ma5 = float(close.rolling(window=5).mean().iloc[-1])
    ma20 = float(close.rolling(window=20).mean().iloc[-1])
    ma60 = float(close.rolling(window=60).mean().iloc[-1])

    # ---------- 技術面評分 ----------
    score = 50  # 基礎分

    # RSI 區間
    if current_rsi < 30:
        score += 10  # 超賣
    elif current_rsi > 70:
        score -= 10  # 超買

    # 均線多頭排列
    if ma5 > ma20:
        score += 10
    if ma20 > ma60:
        score += 10

    # 60 日趨勢
    if len(close) >= 60:
        ret_60 = (close.iloc[-1] - close.iloc[-60]) / close.iloc[-60]
        score += float(ret_60 * 50)  # 趨勢強度影響分數

    # 波動度
    vol = returns.std() * np.sqrt(252)
    if vol < 0.2:
        score += 5
    elif vol > 0.4:
        score -= 5

    technical_score = int(np.clip(score, 0, 100))

    # ---------- 其它三面向（目前先用簡化設定，可日後串財報 & 籌碼 & 新聞） ----------
    fundamental_score = 70
    chip_score = 65
    news_score = 60

    # ---------- 綜合 AI 評分 ----------
    ai_score = int(
        np.clip(
            (technical_score + fundamental_score + chip_score + news_score) / 4.0,
            0,
            100,
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

    # ---------- ROI 估計 ----------
    roi_1d = compute_horizon_roi(close, 1)
    roi_5d = compute_horizon_roi(close, 5)
    roi_60d = compute_horizon_roi(close, 60)
    roi_1y = compute_horizon_roi(close, 252)  # 252 交易日 ≒ 1 年

    roi_dict = {
        "day_1": round(roi_1d, 2),
        "day_5": round(roi_5d, 2),
        "day_60": round(roi_60d, 2),
        "day_365": round(roi_1y, 2),
    }

    # ---------- VaR 95% ----------
    if returns.empty:
        var_95 = 0.0
    else:
        var_95 = float(returns.quantile(0.05))
    var_95_percent = var_95 * 100.0

    current_price = float(close.iloc[-1])
    pessimistic_price = current_price * (1 + var_95)

    # ---------- 未來價格預測（線性回歸） ----------
    window = min(120, len(close))
    X = np.arange(window).reshape(-1, 1)
    y = close.iloc[-window:].values
    model = LinearRegression()
    model.fit(X, y)

    future_days = 60
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

    # ---------- 歷史走勢資料（給圖用） ----------
    history_window = min(300, len(close))
    history_data = [
        {"date": d.strftime("%Y-%m-%d"), "price": round(float(p), 2)}
        for d, p in zip(df.index[-history_window:], close[-history_window:])
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


# =====================================
# K 線詳細分析用：技術指標 & 型態偵測
# =====================================

def calc_macd(series: pd.Series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def calc_rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_kd(df: pd.DataFrame, period: int = 9, smooth_k: int = 3, smooth_d: int = 3):
    low_min = df["Low"].rolling(window=period).min()
    high_max = df["High"].rolling(window=period).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=smooth_k - 1, adjust=False).mean()
    d = k.ewm(com=smooth_d - 1, adjust=False).mean()
    return k, d


def calc_bbands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + num_std * std
    lower = ma - num_std * std
    return upper, ma, lower


def detect_candle_patterns(df: pd.DataFrame) -> List[dict]:
    """
    簡化版 K 線型態偵測：
    - Hammer（錘子線）
    - Shooting Star（射擊之星）
    - Doji（十字線）
    - Bullish Engulfing（多頭吞噬）
    - Bearish Engulfing（空頭吞噬）
    回傳格式：[{date, pattern}, ...]
    """
    patterns = []
    o = df["Open"]
    h = df["High"]
    l = df["Low"]
    c = df["Close"]

    for i in range(1, len(df)):
        date_str = df.index[i].strftime("%Y-%m-%d")
        body = abs(c.iloc[i] - o.iloc[i])
        total_range = h.iloc[i] - l.iloc[i]
        if total_range == 0:
            continue
        upper_shadow = h.iloc[i] - max(c.iloc[i], o.iloc[i])
        lower_shadow = min(c.iloc[i], o.iloc[i]) - l.iloc[i]

        # Doji：實體很小
        if body <= total_range * 0.1:
            patterns.append({"date": date_str, "pattern": "Doji 十字線"})
            continue

        # Hammer：下影線長，上影線短，實體在上方
        if (
            lower_shadow >= body * 2
            and upper_shadow <= body * 0.5
            and (c.iloc[i] > o.iloc[i])
        ):
            patterns.append({"date": date_str, "pattern": "Hammer 錘子線"})
            continue

        # Shooting Star：上影線長，下影線短，實體在下方
        if (
            upper_shadow >= body * 2
            and lower_shadow <= body * 0.5
            and (c.iloc[i] < o.iloc[i])
        ):
            patterns.append({"date": date_str, "pattern": "Shooting Star 射擊之星"})
            continue

        # Bullish Engulfing：多頭吞噬
        prev_body = abs(c.iloc[i - 1] - o.iloc[i - 1])
        if (
            c.iloc[i] > o.iloc[i]
            and c.iloc[i - 1] < o.iloc[i - 1]
            and o.iloc[i] < c.iloc[i - 1]
            and c.iloc[i] > o.iloc[i - 1]
        ):
            patterns.append({"date": date_str, "pattern": "Bullish Engulfing 多頭吞噬"})
            continue

        # Bearish Engulfing：空頭吞噬
        if (
            c.iloc[i] < o.iloc[i]
            and c.iloc[i - 1] > o.iloc[i - 1]
            and o.iloc[i] > c.iloc[i - 1]
            and c.iloc[i] < o.iloc[i - 1]
        ):
            patterns.append({"date": date_str, "pattern": "Bearish Engulfing 空頭吞噬"})
            continue

    return patterns


# =====================================
# API：登入 / 註冊
# =====================================

@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (form_data.username,))
    user_row = c.fetchone()
    conn.close()

    if not user_row or not verify_password(
        form_data.password, user_row["hashed_password"]
    ):
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


# =====================================
# API：核心分析 / 戰情室資料
# =====================================

@app.post("/api/analyze")
async def analyze_stock(request: AnalysisRequest):
    try:
        ticker = yf.Ticker(request.symbol)
        df = ticker.history(period="5y")

        if df.empty:
            raise HTTPException(status_code=404, detail="Stock not found")

        ai_result = calculate_ai_metrics(df, request.symbol)
        if not ai_result:
            raise HTTPException(status_code=400, detail="Not enough data for analysis")

        current_price = ai_result["current_price"]
        principal = request.principal

        # 資金配置
        max_shares = int(principal // current_price) if current_price > 0 else 0
        cost = max_shares * current_price

        # 波段建議
        buy_price = current_price
        take_profit = buy_price * 1.20
        stop_loss = buy_price * 0.90

        # ROI 金額（以歷史平均報酬率估計）
        roi_pct = ai_result["roi"]
        roi_day_amt = cost * roi_pct["day_1"] / 100.0
        roi_week_amt = cost * roi_pct["day_5"] / 100.0
        roi_60_amt = cost * roi_pct["day_60"] / 100.0
        roi_year_amt = cost * roi_pct["day_365"] / 100.0

        # 極端行情風險估計
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
            # 技術指標
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
            # ROI 模組（% + 金額）
            "roi_estimates": {
                "day": {"amt": round(roi_day_amt, 2), "pct": roi_pct["day_1"]},
                "week": {"amt": round(roi_week_amt, 2), "pct": roi_pct["day_5"]},
                "month": {"amt": round(roi_60_amt, 2), "pct": roi_pct["day_60"]},
                "year": {"amt": round(roi_year_amt, 2), "pct": roi_pct["day_365"]},
            },
            # 極端行情預警
            "risk_analysis": {
                "max_drawdown_pct": ai_result["var_95_percent"],
                "max_loss_amt": round(max_loss_amt, 2),
                "pessimistic_price": ai_result["pessimistic_price"],
            },
            # 圖表資料（歷史 + 預測）
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


# =====================================
# API：全球市場快訊（假資料版）
# =====================================

@app.get("/api/news")
async def get_news():
    return [
        {
            "time": "剛剛",
            "title": "半導體庫存去化順利，下半年展望樂觀",
            "source": "產業",
        },
        {
            "time": "5 分鐘前",
            "title": "電動車市場競爭白熱化，車廠降價搶市佔",
            "source": "產業",
        },
        {
            "time": "20 分鐘前",
            "title": "中東地緣政治緊張，油價波動加劇",
            "source": "風險",
        },
    ]


# =====================================
# API：模擬資產管理（需登入）
# =====================================

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
        # 簡化：現價 = 成本 * 1.05
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
async def add_to_portfolio(
    item: PortfolioItem, user: str = Depends(get_current_user)
):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO portfolio (username, symbol, shares, avg_cost) VALUES (?, ?, ?, ?)",
        (user, item.symbol, item.shares, item.cost),
    )
    conn.commit()
    conn.close()
    return {"message": "Added to portfolio"}


# =====================================
# API：收藏股票（需登入）
# =====================================

@app.get("/api/favorites")
async def get_favorites(user: str = Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT symbol FROM favorites WHERE username=?", (user,))
    rows = c.fetchall()
    conn.close()
    symbols = [row["symbol"] for row in rows]
    return {"favorites": symbols}


@app.post("/api/favorites/add")
async def add_favorite(
    item: FavoriteItem, user: str = Depends(get_current_user)
):
    conn = get_db()
    c = conn.cursor()
    # 檢查是否已存在
    c.execute(
        "SELECT id FROM favorites WHERE username=? AND symbol=?",
        (user, item.symbol),
    )
    row = c.fetchone()
    if row is None:
        c.execute(
            "INSERT INTO favorites (username, symbol) VALUES (?, ?)",
            (user, item.symbol),
        )
        conn.commit()
    conn.close()
    return {"message": "Added to favorites"}


@app.post("/api/favorites/remove")
async def remove_favorite(
    item: FavoriteItem, user: str = Depends(get_current_user)
):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        "DELETE FROM favorites WHERE username=? AND symbol=?",
        (user, item.symbol),
    )
    conn.commit()
    conn.close()
    return {"message": "Removed from favorites"}


# =====================================
# API：K 線詳細分析（需登入）
# =====================================

@app.get("/api/kline-detail")
async def kline_detail(
    symbol: str = Query(..., description="股票代碼，如 2330.TW"),
    interval: str = Query("1d", description="K 線週期：1d / 1wk / 1mo"),
    user: str = Depends(get_current_user),
):
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2y", interval=interval)
        if df.empty:
            raise HTTPException(status_code=404, detail="Stock not found")

        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        df = df.sort_index()

        # 最近 200 根即可
        if len(df) > 200:
            df = df.iloc[-200:]

        # 技術指標
        close = df["Close"]
        macd, macd_signal, macd_hist = calc_macd(close)
        rsi = calc_rsi(close)
        k, d = calc_kd(df)
        upper, mid, lower = calc_bbands(close)

        # 型態偵測
        patterns = detect_candle_patterns(df)

        candles = [
            {
                "date": idx.strftime("%Y-%m-%d"),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            }
            for idx, row in df.iterrows()
        ]

        indicators = {
            "ma5": [
                {"date": idx.strftime("%Y-%m-%d"), "value": float(v)}
                for idx, v in close.rolling(window=5).mean().items()
                if not np.isnan(v)
            ],
            "ma20": [
                {"date": idx.strftime("%Y-%m-%d"), "value": float(v)}
                for idx, v in close.rolling(window=20).mean().items()
                if not np.isnan(v)
            ],
            "ma60": [
                {"date": idx.strftime("%Y-%m-%d"), "value": float(v)}
                for idx, v in close.rolling(window=60).mean().items()
                if not np.isnan(v)
            ],
            "macd": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "macd": float(m),
                    "signal": float(s),
                    "hist": float(h),
                }
                for (idx, m), s, h in zip(
                    macd.items(), macd_signal.values, macd_hist.values
                )
                if not np.isnan(m)
            ],
            "rsi": [
                {"date": idx.strftime("%Y-%m-%d"), "value": float(v)}
                for idx, v in rsi.items()
                if not np.isnan(v)
            ],
            "kd": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "k": float(kv),
                    "d": float(dv),
                }
                for (idx, kv), dv in zip(k.items(), d.values)
                if not np.isnan(kv) and not np.isnan(dv)
            ],
            "bbands": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "upper": float(u),
                    "middle": float(m),
                    "lower": float(lw),
                }
                for (idx, u), m, lw in zip(
                    upper.items(), mid.values, lower.values
                )
                if not np.isnan(u)
            ],
        }

        return {
            "symbol": symbol.upper(),
            "interval": interval,
            "candles": candles,
            "indicators": indicators,
            "patterns": patterns,
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in kline_detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================
# 啟動（本機開發用）
# =====================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
