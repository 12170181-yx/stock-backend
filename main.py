# ===============================
# 檔案：stock-backend/main.py
# 目的：強化登入/註冊、修正 API 可上線、補齊收藏/新聞/K線詳細分析(需登入)
# FastAPI + SQLite + JWT + yfinance
# ===============================

import os
import re
import time
import sqlite3
import datetime
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from passlib.context import CryptContext
from jose import JWTError, jwt


# -----------------------------
# 基本設定（可用環境變數覆蓋）
# -----------------------------
APP_NAME = os.getenv("APP_NAME", "stock-backend")
DB_PATH = os.getenv("DB_PATH", "stock_app.db")

# ⚠️ 上線務必改成環境變數（Render / Vercel）
SECRET_KEY = os.getenv("SECRET_KEY", "your_secret_key_here_please_change")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

# CORS（可用逗號分隔多個）
# 建議至少填入你的 Vercel 網址：https://stock-frontend-theta.vercel.app
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
if cors_origins_env.strip() == "*":
    CORS_ORIGINS = ["*"]
else:
    CORS_ORIGINS = [o.strip() for o in cors_origins_env.split(",") if o.strip()]


# -----------------------------
# App 初始化
# -----------------------------
app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# -----------------------------
# 帳密規則（後端也要驗證）
# -----------------------------
USERNAME_REGEX = re.compile(r"^[A-Za-z0-9_]{4,20}$")


def validate_username(username: str) -> None:
    if not USERNAME_REGEX.match(username or ""):
        raise HTTPException(
            status_code=400,
            detail="帳號格式不正確：需 4–20 碼，且僅能包含英文、數字、底線（_）"
        )


def validate_password(password: str) -> None:
    if password is None:
        raise HTTPException(status_code=400, detail="密碼不可為空")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="密碼格式不正確：至少 8 碼")
    if not re.search(r"[A-Za-z]", password):
        raise HTTPException(status_code=400, detail="密碼格式不正確：需包含至少 1 個英文字母")
    if not re.search(r"[0-9]", password):
        raise HTTPException(status_code=400, detail="密碼格式不正確：需包含至少 1 個數字")


# -----------------------------
# DB 初始化
# -----------------------------
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    c = conn.cursor()

    # users
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            hashed_password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    # portfolio
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            symbol TEXT NOT NULL,
            shares INTEGER NOT NULL,
            avg_cost REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(username) REFERENCES users(username)
        )
        """
    )

    # favorites
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            symbol TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(username) REFERENCES users(username)
        )
        """
    )

    # favorites unique index（同一使用者同一股票不可重複收藏）
    c.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_fav_unique
        ON favorites(username, symbol)
        """
    )

    conn.commit()
    conn.close()


init_db()


# -----------------------------
# Pydantic Models
# -----------------------------
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
    duration: str  # "當沖(1日)" / "短期(5日)" / "中期(60日)" / "長期(1年)"


class PortfolioItem(BaseModel):
    symbol: str
    shares: int
    cost: float


class FavoriteItem(BaseModel):
    symbol: str


# -----------------------------
# Auth Helpers
# -----------------------------
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="尚未登入或登入已過期，請重新登入",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if not username:
            raise credentials_exception
        return username
    except JWTError:
        raise credentials_exception


# -----------------------------
# 小工具：時間戳 → 幾分鐘前
# -----------------------------
def time_ago(ts: int) -> str:
    # ts: unix seconds
    now = int(time.time())
    diff = max(0, now - int(ts))
    if diff < 60:
        return "剛剛"
    if diff < 3600:
        return f"{diff // 60} 分鐘前"
    if diff < 86400:
        return f"{diff // 3600} 小時前"
    return f"{diff // 86400} 天前"


# -----------------------------
# 技術指標（不依賴 TA-Lib）
# -----------------------------
def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(50)


def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_macd(close: pd.Series) -> Dict[str, pd.Series]:
    ema12 = calc_ema(close, 12)
    ema26 = calc_ema(close, 26)
    macd = ema12 - ema26
    signal = calc_ema(macd, 9)
    hist = macd - signal
    return {"macd": macd, "signal": signal, "hist": hist}


def calc_kd(df: pd.DataFrame, k_period: int = 9, d_period: int = 3) -> Dict[str, pd.Series]:
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k = rsv.rolling(d_period).mean()
    d = k.rolling(d_period).mean()
    return {"k": k.fillna(method="bfill").fillna(50), "d": d.fillna(method="bfill").fillna(50)}


def calc_bollinger(close: pd.Series, period: int = 20, n_std: float = 2.0) -> Dict[str, pd.Series]:
    ma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return {"mid": ma, "upper": upper, "lower": lower}


# -----------------------------
# K線形態（先做常見偵測 + 48 型態清單供前端顯示）
# -----------------------------
CANDLE_PATTERN_48 = [
    # 這裡提供「名稱清單」，前端要展示 48 種型態用
    "錘子線(Hammer)", "倒錘子線(Inverted Hammer)", "上吊線(Hanging Man)", "流星線(Shooting Star)",
    "吞噬(Engulfing)", "穿頭破腳(Piercing / Dark Cloud Cover)", "晨星(Morning Star)", "夜星(Evening Star)",
    "十字線(Doji)", "墓碑十字(Gravestone Doji)", "蜻蜓十字(Dragonfly Doji)", "長腳十字(Long-Legged Doji)",
    "紅三兵(Three White Soldiers)", "黑三鴉(Three Black Crows)", "上升三法(Rising Three Methods)", "下降三法(Falling Three Methods)",
    "孕線(Harami)", "十字孕線(Harami Cross)", "刺透線(Piercing Line)", "烏雲蓋頂(Dark Cloud Cover)",
    "三內升(Three Inside Up)", "三內降(Three Inside Down)", "三外升(Three Outside Up)", "三外降(Three Outside Down)",
    "跳空缺口(Gap)", "島型反轉(Island Reversal)", "塔形頂/底(Tower Top/Bottom)", "捉腰帶(Belt Hold)",
    "踢腳線(Kicking)", "夾擊線(Meeting Lines)", "分離線(Separating Lines)", "斬回線(Thrusting)",
    "倒錘反轉(Inverted Hammer Reversal)", "錘反轉(Hammer Reversal)", "長紅/長黑(Strong Body)", "紡錘線(Spinning Top)",
    "內包(Inside Bar)", "外包(Outside Bar)", "長上影(Long Upper Shadow)", "長下影(Long Lower Shadow)",
    "高浪線(High Wave)", "三線打擊(Three-Line Strike)", "棄嬰(Abandoned Baby)", "反轉十字(Reversal Doji)",
    "盤整突破(Consolidation Breakout)", "假突破(False Breakout)", "頭肩頂(Head & Shoulders)", "頭肩底(Inverse H&S)"
]


def detect_basic_candle_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    先偵測常用幾種：Doji, Hammer, Engulfing, Morning Star（可後續擴充）
    回傳：[{date, pattern, direction, confidence}]
    """
    out = []
    if len(df) < 5:
        return out

    d = df.copy().tail(60)

    for i in range(2, len(d)):
        row = d.iloc[i]
        prev = d.iloc[i - 1]
        prev2 = d.iloc[i - 2]

        o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
        po, pc = float(prev["Open"]), float(prev["Close"])

        body = abs(c - o)
        rng = max(1e-9, h - l)
        upper_shadow = h - max(o, c)
        lower_shadow = min(o, c) - l

        date_str = d.index[i].strftime("%Y-%m-%d")

        # Doji：實體很小
        if body / rng < 0.08:
            out.append({"date": date_str, "pattern": "十字線(Doji)", "direction": "中性", "confidence": 0.55})

        # Hammer：下影長、實體小、上影短（偏反轉）
        if (lower_shadow / rng > 0.45) and (upper_shadow / rng < 0.15) and (body / rng < 0.35):
            direction = "多轉" if c >= o else "多轉(弱)"
            out.append({"date": date_str, "pattern": "錘子線(Hammer)", "direction": direction, "confidence": 0.70})

        # Engulfing：吞噬
        # 多頭吞噬：前一根黑K，當天紅K且實體包住前一根實體
        if (pc < po) and (c > o) and (c >= po) and (o <= pc):
            out.append({"date": date_str, "pattern": "吞噬(Engulfing)", "direction": "多轉", "confidence": 0.72})
        # 空頭吞噬：前一根紅K，當天黑K且實體包住前一根實體
        if (pc > po) and (c < o) and (o >= pc) and (c <= po):
            out.append({"date": date_str, "pattern": "吞噬(Engulfing)", "direction": "空轉", "confidence": 0.72})

        # Morning Star（簡化版）：連續三根：黑K → 小實體 → 紅K，且紅K收盤回到第一根實體中段以上
        o2, c2 = float(prev2["Open"]), float(prev2["Close"])
        if (c2 < o2) and (abs(pc - po) / max(1e-9, float(prev["High"] - prev["Low"])) < 0.25) and (c > o):
            mid_first = (o2 + c2) / 2.0
            if c >= mid_first:
                out.append({"date": date_str, "pattern": "晨星(Morning Star)", "direction": "多轉", "confidence": 0.75})

    # 去重（同日同型態）
    uniq = {}
    for x in out:
        key = (x["date"], x["pattern"], x["direction"])
        uniq[key] = x
    return list(uniq.values())


# -----------------------------
# 評分：用「真實數據」計算，確保同一份資料評分一致
# -----------------------------
def clamp(v: float, a: float, b: float) -> float:
    return max(a, min(b, v))


def score_technical(df: pd.DataFrame) -> float:
    close = df["Close"]
    rsi = calc_rsi(close).iloc[-1]
    ma5 = close.rolling(5).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]
    ma60 = close.rolling(60).mean().iloc[-1] if len(df) >= 60 else close.rolling(20).mean().iloc[-1]

    macd_pack = calc_macd(close)
    macd_hist = macd_pack["hist"].iloc[-1]

    # 趨勢分：均線多頭排列
    trend = 0.0
    if ma5 > ma20:
        trend += 15
    if ma20 > ma60:
        trend += 15

    # 動能分：RSI（超買扣、超賣加）
    mom = 0.0
    if rsi < 30:
        mom += 18
    elif rsi < 45:
        mom += 8
    elif rsi > 70:
        mom -= 10
    elif rsi > 60:
        mom -= 4

    # MACD 柱狀體
    macd_score = 0.0
    if macd_hist > 0:
        macd_score += 12
    else:
        macd_score -= 6

    # 波動：用年化波動（越低越穩加分，但太低也不一定）
    ret = close.pct_change().dropna()
    vol = float(ret.std() * np.sqrt(252)) if len(ret) > 10 else 0.3
    vol_score = 0.0
    if vol < 0.20:
        vol_score += 10
    elif vol < 0.30:
        vol_score += 5
    elif vol > 0.45:
        vol_score -= 8

    base = 50.0
    score = base + trend + mom + macd_score + vol_score
    return clamp(score, 0, 100)


def score_fundamental_placeholder(symbol: str) -> float:
    """
    基本面：若要做到真正完整（營收/毛利/ROE/FCF…）
    需要接「台股財報來源」或付費 API。
    目前先給一個「可重現」的 placeholder：用 yfinance 的一些欄位（不保證台股齊全）。
    你後續若指定財報來源，我可以把這裡做成真完整版本。
    """
    try:
        t = yf.Ticker(symbol)
        info = t.info or {}
        pe = info.get("trailingPE", None)
        pb = info.get("priceToBook", None)

        score = 55.0
        if pe is not None:
            if pe < 15:
                score += 10
            elif pe > 35:
                score -= 8
        if pb is not None:
            if pb < 2:
                score += 6
            elif pb > 6:
                score -= 6

        return clamp(score, 0, 100)
    except Exception:
        return 55.0


def score_chip_placeholder(df: pd.DataFrame) -> float:
    """
    籌碼面：台股法人/融資融券/大戶…需要專門資料源。
    先用「量能趨勢」做可重現的近似：量增價漲偏多、量增價跌偏空。
    """
    if "Volume" not in df.columns or len(df) < 30:
        return 55.0

    close = df["Close"]
    vol = df["Volume"]

    vol20 = vol.rolling(20).mean().iloc[-1]
    vol5 = vol.rolling(5).mean().iloc[-1]
    ret5 = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0.0

    score = 55.0
    if vol5 > vol20 * 1.2 and ret5 > 0:
        score += 12
    if vol5 > vol20 * 1.2 and ret5 < 0:
        score -= 10
    return clamp(score, 0, 100)


def score_news_placeholder(news_items: List[Dict[str, Any]]) -> float:
    """
    消息面：真正情緒分析需要 NLP。
    先用「新聞數量」做可重現近似：新聞多＝事件多，給中性略加權。
    """
    n = len(news_items)
    score = 55.0
    if n >= 10:
        score += 8
    elif n >= 5:
        score += 4
    return clamp(score, 0, 100)


def calc_roi_estimates(principal_cost: float, close: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    ROI：用歷史日報酬的平均 + 波動做估計（可重現、依據真實資料）
    """
    r = close.pct_change().dropna()
    if len(r) < 50:
        # fallback
        day_mu = 0.001
        day_sigma = 0.02
    else:
        day_mu = float(r.mean())
        day_sigma = float(r.std())

    def est(days: int) -> Dict[str, float]:
        # 期望報酬（線性近似）
        exp_ret = day_mu * days
        # 風險（簡化：sigma*sqrt(days)）
        risk = day_sigma * np.sqrt(days)
        # 避免太誇張（限制）
        exp_ret = float(clamp(exp_ret, -0.35, 0.80))
        amt = principal_cost * exp_ret
        return {"pct": round(exp_ret * 100, 2), "amt": round(amt, 0)}

    return {
        "day": est(1),
        "short": est(5),
        "mid": est(60),
        "long": est(252),
    }


def calc_var95(close: pd.Series, horizon_days: int = 60) -> Dict[str, float]:
    """
    極端行情預警：用歷史日報酬分位數近似 VaR（可重現）
    這裡回傳「60 天」極端跌幅估計與悲觀目標價
    """
    r = close.pct_change().dropna()
    if len(r) < 50:
        q05 = -0.03
        sigma = 0.02
    else:
        q05 = float(r.quantile(0.05))  # 單日 5% 分位
        sigma = float(r.std())

    # 將單日分位放大到 60 天（簡化：乘 sqrt）
    var_h = q05 * np.sqrt(horizon_days)
    var_h = float(clamp(var_h, -0.60, 0.0))

    current_price = float(close.iloc[-1])
    pessimistic_price = current_price * (1 + var_h)

    return {
        "var_h_pct": round(var_h * 100, 2),
        "pessimistic_price": round(pessimistic_price, 2),
    }


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
async def health():
    return {"ok": True, "service": APP_NAME, "time": datetime.datetime.utcnow().isoformat()}


@app.post("/register")
async def register(user: User):
    u = (user.username or "").strip()
    p = user.password or ""

    validate_username(u)
    validate_password(p)

    conn = get_db()
    c = conn.cursor()

    # 檢查是否存在
    c.execute("SELECT username FROM users WHERE username=?", (u,))
    if c.fetchone():
        conn.close()
        raise HTTPException(status_code=400, detail="此帳號已存在，請更換帳號")

    hashed_pw = get_password_hash(p)
    now = datetime.datetime.utcnow().isoformat()

    c.execute(
        "INSERT INTO users (username, hashed_password, created_at) VALUES (?, ?, ?)",
        (u, hashed_pw, now)
    )
    conn.commit()
    conn.close()

    return {"message": "User created successfully"}


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    u = (form_data.username or "").strip()
    p = form_data.password or ""

    # 這裡不強制 validate_password（避免使用者舊密碼格式被擋），只驗證帳號格式
    validate_username(u)

    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT username, hashed_password FROM users WHERE username=?", (u,))
    row = c.fetchone()
    conn.close()

    if (not row) or (not verify_password(p, row["hashed_password"])):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": row["username"]})
    return {"access_token": access_token, "token_type": "bearer"}


# -----------------------------
# 真實新聞（可點擊連結）
# - /api/news?symbol=2330.TW → 取該股新聞
# - /api/news → 取全球新聞（用大盤/科技股作近似）
# -----------------------------
@app.get("/api/news")
async def get_news(symbol: Optional[str] = Query(default=None, description="可選：股票代碼，例如 2330.TW")):
    try:
        if symbol:
            t = yf.Ticker(symbol.strip())
        else:
            # global：用 ^GSPC 或 AAPL 取近似「全球市場快訊」
            t = yf.Ticker("^GSPC")

        items = t.news or []
        out = []

        for it in items[:20]:
            # yfinance news 常見欄位：title, link, publisher, providerPublishTime
            title = it.get("title") or ""
            link = it.get("link") or it.get("url") or ""
            publisher = it.get("publisher") or it.get("source") or "News"
            ts = it.get("providerPublishTime") or int(time.time())
            out.append({
                "tag": "市場" if not symbol else "個股",
                "time": time_ago(int(ts)),
                "title": title,
                "source": publisher,
                "url": link
            })

        # 若抓不到資料，回傳空陣列（前端顯示「目前沒有新聞」）
        return out
    except Exception as e:
        # 不要讓新聞壞掉拖垮整個頁面
        return []


# -----------------------------
# 收藏（需登入）
# -----------------------------
@app.get("/api/favorites")
async def list_favorites(user: str = Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT symbol, created_at FROM favorites WHERE username=? ORDER BY created_at DESC", (user,))
    rows = c.fetchall()
    conn.close()

    return [{"symbol": r["symbol"], "created_at": r["created_at"]} for r in rows]


@app.post("/api/favorites/add")
async def add_favorite(item: FavoriteItem, user: str = Depends(get_current_user)):
    symbol = (item.symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol 不可為空")

    conn = get_db()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO favorites (username, symbol, created_at) VALUES (?, ?, ?)",
            (user, symbol, datetime.datetime.utcnow().isoformat())
        )
        conn.commit()
    except sqlite3.IntegrityError:
        # 已存在就視為成功（避免前端重按爆炸）
        pass
    finally:
        conn.close()

    return {"ok": True, "symbol": symbol}


@app.post("/api/favorites/remove")
async def remove_favorite(item: FavoriteItem, user: str = Depends(get_current_user)):
    symbol = (item.symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol 不可為空")

    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM favorites WHERE username=? AND symbol=?", (user, symbol))
    conn.commit()
    conn.close()

    return {"ok": True, "symbol": symbol}


# -----------------------------
# 模擬資產（需登入） - 保留你原本功能，但改成「會抓真實現價」更合理
# -----------------------------
@app.get("/api/portfolio")
async def get_portfolio(user: str = Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT * FROM portfolio WHERE username=?", (user,))
    rows = c.fetchall()
    conn.close()

    holdings = []
    total_asset = 0.0
    total_cost = 0.0

    # 逐一抓現價（簡單版）
    for row in rows:
        symbol = row["symbol"]
        shares = int(row["shares"])
        avg_cost = float(row["avg_cost"])

        current_price = avg_cost
        try:
            t = yf.Ticker(symbol)
            df = t.history(period="5d")
            if not df.empty:
                current_price = float(df["Close"].iloc[-1])
        except Exception:
            pass

        cost = shares * avg_cost
        value = shares * current_price
        pnl = value - cost

        holdings.append({
            "symbol": symbol,
            "shares": shares,
            "cost": round(avg_cost, 2),
            "current_price": round(current_price, 2),
            "market_value": round(value, 0),
            "pnl": round(pnl, 0),
        })

        total_asset += value
        total_cost += cost

    return {
        "total_asset": round(total_asset, 0),
        "total_cost": round(total_cost, 0),
        "unrealized_pnl": round(total_asset - total_cost, 0),
        "holdings": holdings
    }


@app.post("/api/portfolio/add")
async def add_to_portfolio(item: PortfolioItem, user: str = Depends(get_current_user)):
    symbol = (item.symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol 不可為空")
    if item.shares <= 0:
        raise HTTPException(status_code=400, detail="shares 必須大於 0")
    if item.cost <= 0:
        raise HTTPException(status_code=400, detail="cost 必須大於 0")

    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO portfolio (username, symbol, shares, avg_cost, created_at) VALUES (?, ?, ?, ?, ?)",
        (user, symbol, int(item.shares), float(item.cost), datetime.datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

    return {"message": "Added to portfolio"}


# -----------------------------
# 核心分析（/api/analyze）
# - 回傳：AI 綜合評分 + ROI + 波段價位 + 極端行情預警 + 圖表資料
# -----------------------------
@app.post("/api/analyze")
async def analyze_stock(request: AnalysisRequest):
    symbol = (request.symbol or "").strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol 不可為空")
    if request.principal <= 0:
        raise HTTPException(status_code=400, detail="principal 必須大於 0")

    # 取 2 年資料：更穩
    try:
        t = yf.Ticker(symbol)
        df = t.history(period="2y", auto_adjust=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"抓取股價資料失敗：{str(e)}")

    if df is None or df.empty or len(df) < 80:
        raise HTTPException(status_code=400, detail="資料不足（至少需要約 80 個交易日資料）")

    # 技術指標 / 評分
    tech_score = float(score_technical(df))
    # 新聞（可用於消息面分數）
    news_items = []
    try:
        items = (t.news or [])[:20]
        for it in items:
            news_items.append(it)
    except Exception:
        news_items = []

    fund_score = float(score_fundamental_placeholder(symbol))
    chip_score = float(score_chip_placeholder(df))
    news_score = float(score_news_placeholder(news_items))

    # 綜合評分（四大面向加總平均，確保可重現）
    ai_score = round((tech_score + fund_score + chip_score + news_score) / 4.0, 0)

    if ai_score >= 80:
        sentiment = "強力看多"
    elif ai_score >= 60:
        sentiment = "偏多"
    elif ai_score >= 40:
        sentiment = "中立"
    else:
        sentiment = "偏空"

    close = df["Close"].copy()
    current_price = float(close.iloc[-1])

    # 資金配置
    max_shares = int(request.principal // current_price)
    total_cost = float(max_shares * current_price)
    risk_loss_10 = float(total_cost * 0.10)

    # 波段價位（先用現價作買入，停利停損固定比例；後續可改用 ATR/支撐壓力）
    buy_price = current_price
    take_profit = round(buy_price * 1.20, 2)
    stop_loss = round(buy_price * 0.90, 2)

    # ROI 估算（用真實歷史報酬）
    roi = calc_roi_estimates(total_cost, close)

    # 極端行情預警（60 天）
    var_pack = calc_var95(close, horizon_days=60)
    max_loss_amt = round(total_cost * abs(var_pack["var_h_pct"]) / 100.0, 0)

    # 圖表資料：回最近 120 天
    hist_days = 120
    hist_df = df.tail(hist_days)
    history = []
    for idx, row in hist_df.iterrows():
        history.append({
            "date": idx.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0
        })

    return {
        "symbol": symbol.upper(),
        "price": round(current_price, 2),

        # 核心分數
        "ai_score": int(ai_score),
        "ai_sentiment": sentiment,

        # 四大面向（前端可直接顯示）
        "score_breakdown": {
            "technical": int(round(tech_score, 0)),
            "fundamental": int(round(fund_score, 0)),
            "chip": int(round(chip_score, 0)),
            "news": int(round(news_score, 0)),
        },

        # 資金配置試算
        "money_management": {
            "principal": float(request.principal),
            "max_shares": max_shares,
            "total_cost": round(total_cost, 2),
            "risk_loss_10_percent": round(risk_loss_10, 2),
        },

        # 波段建議
        "advice": {
            "buy_price": round(buy_price, 2),
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        },

        # ROI 預估（四個期限）
        "roi_estimates": {
            "day": roi["day"],
            "short": roi["short"],
            "mid": roi["mid"],
            "long": roi["long"],
        },

        # 風險（極端行情預警）
        "risk_analysis": {
            "max_loss_pct_60d": var_pack["var_h_pct"],
            "max_loss_amt_60d": float(max_loss_amt),
            "pessimistic_price_60d": var_pack["pessimistic_price"],
            "confidence_level": "95%",
        },

        # 圖表資料（K線用）
        "chart_data": {
            "history": history
        }
    }


# -----------------------------
# K 線詳細分析（需登入）
# - 支援 timeframe：1d, 1wk, 1mo, 1h, 30m, 15m, 5m...
# -----------------------------
@app.get("/api/kline-detail")
async def kline_detail(
    symbol: str = Query(..., description="股票代碼，例如 2330.TW"),
    interval: str = Query("1d", description="時間週期：1d/1wk/1mo/1h/30m/15m/5m..."),
    period: str = Query("6mo", description="回溯期間：1mo/3mo/6mo/1y/2y...（分K建議用較短）"),
    user: str = Depends(get_current_user),
):
    s = (symbol or "").strip()
    if not s:
        raise HTTPException(status_code=400, detail="symbol 不可為空")

    try:
        t = yf.Ticker(s)
        df = t.history(period=period, interval=interval, auto_adjust=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"抓取K線資料失敗：{str(e)}")

    if df is None or df.empty or len(df) < 50:
        raise HTTPException(status_code=400, detail="K線資料不足（請換更長 period 或更大 interval）")

    df = df.dropna()
    close = df["Close"]

    # 技術指標
    rsi_series = calc_rsi(close)
    macd_pack = calc_macd(close)
    kd_pack = calc_kd(df)
    bb = calc_bollinger(close)

    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean() if len(df) >= 60 else close.rolling(20).mean()

    # 量能分析
    vol = df["Volume"] if "Volume" in df.columns else pd.Series([0] * len(df), index=df.index)
    vol5 = vol.rolling(5).mean()
    vol20 = vol.rolling(20).mean()

    # K線形態偵測（先做常用）
    patterns = detect_basic_candle_patterns(df)

    # 組合資料給前端
    tail_n = 120 if interval in ["1d", "1wk", "1mo"] else 200
    dft = df.tail(tail_n)

    candles = []
    for idx, row in dft.iterrows():
        candles.append({
            "t": idx.isoformat(),
            "open": round(float(row["Open"]), 4),
            "high": round(float(row["High"]), 4),
            "low": round(float(row["Low"]), 4),
            "close": round(float(row["Close"]), 4),
            "volume": int(row["Volume"]) if not pd.isna(row["Volume"]) else 0,
        })

    indicators = {
        "ma5": round(float(ma5.iloc[-1]), 4) if not pd.isna(ma5.iloc[-1]) else None,
        "ma20": round(float(ma20.iloc[-1]), 4) if not pd.isna(ma20.iloc[-1]) else None,
        "ma60": round(float(ma60.iloc[-1]), 4) if not pd.isna(ma60.iloc[-1]) else None,

        "rsi14": round(float(rsi_series.iloc[-1]), 2),

        "macd": round(float(macd_pack["macd"].iloc[-1]), 4),
        "macd_signal": round(float(macd_pack["signal"].iloc[-1]), 4),
        "macd_hist": round(float(macd_pack["hist"].iloc[-1]), 4),

        "k": round(float(kd_pack["k"].iloc[-1]), 2),
        "d": round(float(kd_pack["d"].iloc[-1]), 2),

        "bb_mid": round(float(bb["mid"].iloc[-1]), 4) if not pd.isna(bb["mid"].iloc[-1]) else None,
        "bb_upper": round(float(bb["upper"].iloc[-1]), 4) if not pd.isna(bb["upper"].iloc[-1]) else None,
        "bb_lower": round(float(bb["lower"].iloc[-1]), 4) if not pd.isna(bb["lower"].iloc[-1]) else None,

        "vol5_avg": round(float(vol5.iloc[-1]), 2) if not pd.isna(vol5.iloc[-1]) else None,
        "vol20_avg": round(float(vol20.iloc[-1]), 2) if not pd.isna(vol20.iloc[-1]) else None,
    }

    # 給一段「判讀摘要」（可重現）
    summary = []
    if indicators["ma5"] and indicators["ma20"] and indicators["ma60"]:
        if indicators["ma5"] > indicators["ma20"] > indicators["ma60"]:
            summary.append("均線呈多頭排列（MA5 > MA20 > MA60），趨勢偏多。")
        elif indicators["ma5"] < indicators["ma20"] < indicators["ma60"]:
            summary.append("均線呈空頭排列（MA5 < MA20 < MA60），趨勢偏空。")
        else:
            summary.append("均線未形成明顯排列，趨勢可能盤整或轉折中。")

    if indicators["rsi14"] >= 70:
        summary.append("RSI 進入超買區（>=70），需留意拉回風險。")
    elif indicators["rsi14"] <= 30:
        summary.append("RSI 進入超賣區（<=30），可能出現反彈。")
    else:
        summary.append("RSI 落在中性區間，動能較均衡。")

    if indicators["macd_hist"] > 0:
        summary.append("MACD 柱狀體為正，動能偏多。")
    else:
        summary.append("MACD 柱狀體為負，動能偏弱。")

    if indicators["vol5_avg"] and indicators["vol20_avg"]:
        if indicators["vol5_avg"] > indicators["vol20_avg"] * 1.2:
            summary.append("近5期均量明顯大於20期均量，量能放大。")
        elif indicators["vol5_avg"] < indicators["vol20_avg"] * 0.8:
            summary.append("近5期均量低於20期均量，量能偏縮。")

    return {
        "symbol": s.upper(),
        "interval": interval,
        "period": period,
        "candles": candles,
        "indicators": indicators,
        "detected_patterns": patterns,
        "pattern_catalog_48": CANDLE_PATTERN_48,
        "summary": summary
    }


# -----------------------------
# 本機啟動
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
