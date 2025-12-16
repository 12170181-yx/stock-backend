import os
import re
import time
import sqlite3
import datetime
from typing import Optional, List, Dict, Any
from email.utils import parsedate_to_datetime

import numpy as np
import pandas as pd
import yfinance as yf

from fastapi import FastAPI, HTTPException, Depends, status, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt

# ====== 可選：真實新聞 RSS 解析（下一個 requirements.txt 會補 feedparser）======
try:
    import feedparser  # type: ignore
except Exception:
    feedparser = None


# ==========================
# 0) 設定
# ==========================
APP_NAME = "stock-backend"

# Render / 本機環境變數
DATABASE_PATH = os.getenv("DATABASE_PATH", "stock_app.db")
SECRET_KEY = os.getenv("SECRET_KEY", "PLEASE_CHANGE_THIS_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

# 你前端的正式網址
VERCEL_FRONTEND = os.getenv("VERCEL_FRONTEND", "https://stock-frontend-theta.vercel.app")

# CORS：正式建議指定你的前端網域 + 本機測試
ALLOWED_ORIGINS = list({
    VERCEL_FRONTEND,
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
})

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================
# 1) DB 初始化
# ==========================
def get_db():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    c = conn.cursor()

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            symbol TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(username, symbol)
        )
        """
    )

    c.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            symbol TEXT NOT NULL,
            shares INTEGER NOT NULL,
            avg_cost REAL NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


init_db()


# ==========================
# 2) Pydantic Models
# ==========================
class UserCreate(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class AnalysisRequest(BaseModel):
    symbol: str
    principal: float
    strategy: str
    duration: str  # day/short/mid/long


class FavoriteReq(BaseModel):
    symbol: str


class PortfolioItem(BaseModel):
    symbol: str
    shares: int
    cost: float


# ==========================
# 3) Auth helpers
# ==========================
USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{4,20}$")
PASSWORD_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d).{8,}$")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(subject: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode = {"sub": subject, "exp": expire}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> str:
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise cred_exc
        return username
    except JWTError:
        raise cred_exc


# ==========================
# 4) Data fetch（保證抓到「最新優先」）
# ==========================
def fetch_price_history(symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    使用 yfinance 抓取公開行情資料。
    規則：只要抓到資料，就以資料最後一筆日期作為「最新」依據。
    """
    # download 較穩定
    df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # 清理
    df = df.dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index)
    return df


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(method="bfill").fillna(50)


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return macd, sig, hist


def compute_bollinger(close: pd.Series, window: int = 20, k: float = 2.0):
    ma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = ma + k * std
    lower = ma - k * std
    return ma, upper, lower


def compute_kd(df: pd.DataFrame, k_period: int = 9, d_period: int = 3):
    low_min = df["Low"].rolling(k_period).min()
    high_max = df["High"].rolling(k_period).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k = rsv.ewm(alpha=1 / d_period, adjust=False).mean()
    d = k.ewm(alpha=1 / d_period, adjust=False).mean()
    return k.fillna(50), d.fillna(50)


# ==========================
# 5) 四大面向評分（可重現、同資料同分數）
# ==========================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def score_technical(df: pd.DataFrame) -> Dict[str, Any]:
    close = df["Close"]
    rsi = compute_rsi(close)
    macd, sig, hist = compute_macd(close)
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    bb_ma, bb_up, bb_low = compute_bollinger(close)
    k, d = compute_kd(df)

    # 取最新值
    rsi_v = float(rsi.iloc[-1])
    macd_v = float(macd.iloc[-1])
    sig_v = float(sig.iloc[-1])
    hist_v = float(hist.iloc[-1])
    ma20_v = float(ma20.iloc[-1]) if not np.isnan(ma20.iloc[-1]) else float(close.iloc[-1])
    ma60_v = float(ma60.iloc[-1]) if not np.isnan(ma60.iloc[-1]) else float(close.iloc[-1])
    ma120_v = float(ma120.iloc[-1]) if not np.isnan(ma120.iloc[-1]) else float(close.iloc[-1])
    k_v = float(k.iloc[-1])
    d_v = float(d.iloc[-1])

    price = float(close.iloc[-1])

    # 分數設計：0-100，保證 deterministic（不亂抽樣）
    s = 50.0

    # 趨勢：均線多頭
    if price > ma20_v:
        s += 8
    if ma20_v > ma60_v:
        s += 10
    if ma60_v > ma120_v:
        s += 10

    # 動能：MACD
    if macd_v > sig_v:
        s += 8
    if hist_v > 0:
        s += 4

    # RSI：避免極端
    if rsi_v < 30:
        s += 6  # 超賣可能反彈
    elif rsi_v > 70:
        s -= 6  # 超買回檔風險

    # KD：多空力道
    if k_v > d_v:
        s += 4
    if k_v < 20:
        s += 2
    if k_v > 80:
        s -= 2

    # 波動：太暴衝扣分、適中加分
    ret = close.pct_change().dropna()
    vol = float(ret.tail(60).std() * np.sqrt(252)) if len(ret) >= 30 else 0.35
    if vol < 0.25:
        s += 6
    elif vol > 0.6:
        s -= 6

    s = clamp(s, 0, 100)

    return {
        "score": int(round(s)),
        "rsi": round(rsi_v, 2),
        "macd": round(macd_v, 4),
        "macd_signal": round(sig_v, 4),
        "ma20": round(ma20_v, 2),
        "ma60": round(ma60_v, 2),
        "ma120": round(ma120_v, 2),
        "bb_mid": round(float(bb_ma.iloc[-1]) if not np.isnan(bb_ma.iloc[-1]) else price, 2),
        "bb_upper": round(float(bb_up.iloc[-1]) if not np.isnan(bb_up.iloc[-1]) else price, 2),
        "bb_lower": round(float(bb_low.iloc[-1]) if not np.isnan(bb_low.iloc[-1]) else price, 2),
        "kd_k": round(k_v, 2),
        "kd_d": round(d_v, 2),
        "annual_volatility": round(vol, 4),
    }


def score_fundamental(symbol: str) -> Dict[str, Any]:
    """
    基本面：使用 yfinance 公開資訊（可能部分標的沒有）
    只要資料缺漏，就用中性值，避免亂跳。
    """
    s = 50.0
    info = {}
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        info = {}

    # 常見欄位（可能為 None）
    roe = info.get("returnOnEquity", None)  # e.g. 0.25
    profit_margin = info.get("profitMargins", None)
    rev_g = info.get("revenueGrowth", None)
    earn_g = info.get("earningsGrowth", None)
    debt_to_eq = info.get("debtToEquity", None)
    fwd_pe = info.get("forwardPE", None)
    pb = info.get("priceToBook", None)

    # ROE
    if isinstance(roe, (int, float)):
        if roe >= 0.2:
            s += 12
        elif roe >= 0.1:
            s += 6
        elif roe < 0.05:
            s -= 6

    # 利潤率
    if isinstance(profit_margin, (int, float)):
        if profit_margin >= 0.15:
            s += 8
        elif profit_margin < 0.05:
            s -= 6

    # 成長
    for g in [rev_g, earn_g]:
        if isinstance(g, (int, float)):
            if g >= 0.15:
                s += 6
            elif g < 0:
                s -= 6

    # 負債（越高越扣）
    if isinstance(debt_to_eq, (int, float)):
        if debt_to_eq > 200:
            s -= 10
        elif debt_to_eq > 100:
            s -= 6
        elif debt_to_eq < 50:
            s += 4

    # 估值（非常粗略）
    if isinstance(fwd_pe, (int, float)):
        if fwd_pe < 12:
            s += 5
        elif fwd_pe > 30:
            s -= 5

    if isinstance(pb, (int, float)):
        if pb < 2:
            s += 3
        elif pb > 6:
            s -= 3

    s = clamp(s, 0, 100)

    # 回傳核心欄位（前端可顯示）
    return {
        "score": int(round(s)),
        "roe": roe,
        "profit_margin": profit_margin,
        "revenue_growth": rev_g,
        "earnings_growth": earn_g,
        "debt_to_equity": debt_to_eq,
        "forward_pe": fwd_pe,
        "price_to_book": pb,
    }


def score_chip_proxy(df: pd.DataFrame) -> Dict[str, Any]:
    """
    籌碼面（台股常見三大法人等公開資料不一定有免費 API）
    先用「量價/資金流代理指標」做可重現評分（不亂掰、不假裝法人資料）。
    """
    close = df["Close"]
    vol = df["Volume"].fillna(0)

    # OBV（簡化）
    direction = np.sign(close.diff().fillna(0))
    obv = (direction * vol).cumsum()

    # 近20日 OBV 斜率：資金流入加分
    x = np.arange(len(obv.tail(20))).reshape(-1, 1)
    y = obv.tail(20).values.reshape(-1, 1)
    slope = 0.0
    if len(y) >= 10:
        # 線性回歸斜率（不使用 sklearn，避免依賴）
        x_mean = x.mean()
        y_mean = y.mean()
        slope = float(((x - x_mean) * (y - y_mean)).sum() / (((x - x_mean) ** 2).sum() + 1e-9))

    # 量能：近5日均量 vs 近20日均量
    v5 = float(vol.tail(5).mean())
    v20 = float(vol.tail(20).mean())
    vol_ratio = (v5 / v20) if v20 > 0 else 1.0

    s = 50.0
    if slope > 0:
        s += 12
    else:
        s -= 6

    if vol_ratio >= 1.3:
        s += 10  # 量增
    elif vol_ratio <= 0.7:
        s -= 6   # 量縮

    s = clamp(s, 0, 100)

    return {
        "score": int(round(s)),
        "obv_slope_20d": round(slope, 4),
        "volume_ratio_5v20": round(vol_ratio, 4),
    }


def score_news_sentiment(news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    消息面：用公開新聞標題做「關鍵字情緒」代理評分（可重現）。
    """
    s = 50.0
    pos_kw = ["上調", "成長", "創新高", "強勁", "利多", "看好", "獲利", "大漲", "買超", "調升"]
    neg_kw = ["下調", "衰退", "利空", "大跌", "賣超", "風險", "崩跌", "裁員", "警告", "疲弱"]

    titles = " ".join([(x.get("title") or "") for x in news_items]).lower()

    pos = sum([titles.count(k.lower()) for k in pos_kw])
    neg = sum([titles.count(k.lower()) for k in neg_kw])

    # 簡單可重現：pos/neg 影響分數
    s += (pos * 2.5)
    s -= (neg * 3.0)

    s = clamp(s, 0, 100)
    return {"score": int(round(s)), "pos_hits": pos, "neg_hits": neg}


def composite_score(tech: int, fund: int, chip: int, news: int) -> int:
    """
    綜合分數：固定權重，確保同資料同結果（可重現、可驗證）。
    """
    # 權重可調：你要「技術面」比重高一點，我先給 0.35
    score = 0.35 * tech + 0.25 * fund + 0.20 * chip + 0.20 * news
    return int(round(clamp(score, 0, 100)))


def sentiment_text(score: int) -> str:
    if score >= 80:
        return "強力看多"
    if score >= 60:
        return "偏多"
    if score >= 40:
        return "中立"
    return "偏空"


# ==========================
# 6) ROI / 風險 / 價格區間
# ==========================
def estimate_roi(cost: float, df: pd.DataFrame) -> Dict[str, Any]:
    """
    用歷史報酬均值+波動推估（代理模型，deterministic）
    """
    close = df["Close"]
    ret = close.pct_change().dropna()
    if ret.empty:
        return {
            "day": {"pct": 0.0, "amt": 0},
            "short": {"pct": 0.0, "amt": 0},
            "mid": {"pct": 0.0, "amt": 0},
            "long": {"pct": 0.0, "amt": 0},
        }

    mu = float(ret.tail(120).mean())  # 近120日平均日報酬
    sigma = float(ret.tail(120).std())  # 近120日波動

    def horizon_pct(days: int) -> float:
        # 保守：mu*days，並用波動做合理裁切（避免離譜）
        p = mu * days
        cap = max(0.08, 2.5 * sigma * np.sqrt(days))  # 波動越大 cap 越大
        return float(clamp(p, -cap, cap))

    horizons = {"day": 1, "short": 5, "mid": 60, "long": 252}
    out = {}
    for k, d in horizons.items():
        pct = horizon_pct(d)
        out[k] = {"pct": round(pct * 100, 2), "amt": int(round(cost * pct))}
    return out


def extreme_risk_95(cost: float, df: pd.DataFrame, horizon_days: int = 60) -> Dict[str, Any]:
    """
    極端行情預警：使用歷史單日 VaR 95% + sqrt(horizon) 近似
    """
    close = df["Close"]
    ret = close.pct_change().dropna()
    if ret.empty:
        return {"max_loss_amt": 0, "max_loss_pct": 0.0, "pessimistic_price": float(close.iloc[-1])}

    var1 = float(ret.quantile(0.05))  # 5% 分位（單日）
    # horizon scale
    var_h = var1 * np.sqrt(horizon_days)

    price = float(close.iloc[-1])
    pessimistic_price = price * (1 + var_h)

    max_loss_amt = int(round(cost * abs(var_h)))
    return {
        "max_loss_amt": max_loss_amt,
        "max_loss_pct": round(abs(var_h) * 100, 2),
        "pessimistic_price": round(pessimistic_price, 2),
        "var_1d_pct": round(abs(var1) * 100, 2),
    }


def band_trade_prices(current_price: float) -> Dict[str, float]:
    buy = current_price
    tp = current_price * 1.20
    sl = current_price * 0.90
    return {
        "buy_price": round(buy, 2),
        "take_profit": round(tp, 2),
        "stop_loss": round(sl, 2),
    }


def build_chart_data(df: pd.DataFrame, future_days: int = 30) -> Dict[str, Any]:
    """
    給前端折線/區間帶使用：history + 以 log 線性趨勢做簡單外推 + 區間
    """
    df_use = df.tail(240).copy()
    close = df_use["Close"].astype(float)

    # log regression (deterministic)
    y = np.log(close.values)
    x = np.arange(len(y))

    # 線性回歸（手寫）
    x_mean = x.mean()
    y_mean = y.mean()
    slope = float(((x - x_mean) * (y - y_mean)).sum() / (((x - x_mean) ** 2).sum() + 1e-9))
    intercept = float(y_mean - slope * x_mean)

    # 殘差標準差
    y_hat = intercept + slope * x
    resid = y - y_hat
    resid_std = float(np.std(resid)) if len(resid) > 5 else 0.02

    # 95% 區間（粗略：±1.96*std）
    z = 1.96

    last_date = df_use.index[-1]
    history = [
        {"date": d.strftime("%Y-%m-%d"), "price": round(float(p), 2)}
        for d, p in zip(df_use.index, close.values)
    ]

    pred = []
    for i in range(1, future_days + 1):
        xi = len(y) - 1 + i
        yi = intercept + slope * xi
        mid = float(np.exp(yi))
        up = float(np.exp(yi + z * resid_std))
        lo = float(np.exp(yi - z * resid_std))

        next_date = last_date + datetime.timedelta(days=i)
        pred.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "mid": round(mid, 2),
            "upper": round(up, 2),
            "lower": round(lo, 2),
        })

    return {"history": history, "prediction": pred}


# ==========================
# 7) 真實新聞（可點擊、最新優先）
# ==========================
def parse_dt(entry: dict) -> Optional[datetime.datetime]:
    # feedparser 的 published_parsed
    if entry is None:
        return None
    if entry.get("published_parsed"):
        try:
            return datetime.datetime(*entry["published_parsed"][:6])
        except Exception:
            pass
    # 有些 feed 是 published string
    if entry.get("published"):
        try:
            return parsedate_to_datetime(entry["published"])
        except Exception:
            return None
    return None


def fetch_real_news(query: str, limit: int = 12) -> List[Dict[str, Any]]:
    """
    使用 Google News RSS（公開）聚合新聞。
    注意：RSS 本身是公開資訊，可點擊跳轉原新聞。
    """
    if feedparser is None:
        # 沒裝 feedparser 時，回空（前端會顯示錯誤訊息）
        return []

    # Google News RSS
    # hl=zh-TW & gl=TW & ceid=TW:zh-Hant
    q = query.strip()
    rss_url = (
        "https://news.google.com/rss/search?"
        f"q={q}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    )

    feed = feedparser.parse(rss_url)
    items = []
    for e in feed.entries[:limit * 2]:
        title = getattr(e, "title", "") or ""
        link = getattr(e, "link", "") or ""
        published = getattr(e, "published", "") or ""
        dt = parse_dt(e) or None

        # 推定分類：簡單用關鍵字
        tag = "產業"
        low = title.lower()
        if any(k in low for k in ["風險", "戰", "地緣", "通膨", "升息", "降息", "fed", "利率", "油價"]):
            tag = "風險"
        if any(k in low for k in ["法說", "財報", "營收", "獲利", "eps", "目標價", "評等", "上調", "下調"]):
            tag = "評論"

        items.append({
            "tag": tag,
            "time": published or (dt.strftime("%Y-%m-%d %H:%M") if dt else ""),
            "published_at": dt.isoformat() if dt else "",
            "title": title,
            "url": link,
            "source": "Google News RSS",
        })

    # 去重（同連結）
    seen = set()
    uniq = []
    for it in items:
        u = it.get("url", "")
        if not u:
            continue
        if u in seen:
            continue
        seen.add(u)
        uniq.append(it)

    # 最新優先
    def sort_key(it):
        ts = it.get("published_at") or ""
        return ts

    uniq.sort(key=sort_key, reverse=True)
    return uniq[:limit]


# ==========================
# 8) 基本服務 / Debug
# ==========================
@app.get("/health")
async def health():
    return {
        "service": APP_NAME,
        "status": "ok",
        "server_time_utc": datetime.datetime.utcnow().isoformat(),
        "allowed_origins": ALLOWED_ORIGINS,
    }


@app.get("/debug/data_status")
async def debug_data_status(symbol: str = Query("2330.TW"), limit_news: int = Query(8, ge=1, le=20)):
    """
    ✅ 用來「確認後端是否真的抓到資料」的驗證端點：
    - 價格資料筆數
    - 最新日期
    - 新聞最新時間
    """
    t0 = time.time()
    df = fetch_price_history(symbol, period="1y")
    fetch_sec = round(time.time() - t0, 3)

    if df.empty:
        return {
            "symbol": symbol,
            "price_rows": 0,
            "price_last_date": None,
            "fetch_seconds": fetch_sec,
            "news_count": 0,
            "news_latest_time": None,
            "sources": ["yfinance", "GoogleNewsRSS" if feedparser else "RSS_DISABLED(feedparser_missing)"],
        }

    last_date = df.index[-1].strftime("%Y-%m-%d")
    news = fetch_real_news(symbol, limit=limit_news)
    news_latest = news[0]["time"] if news else None

    return {
        "symbol": symbol,
        "price_rows": int(len(df)),
        "price_last_date": last_date,
        "fetch_seconds": fetch_sec,
        "news_count": len(news),
        "news_latest_time": news_latest,
        "sources": ["yfinance", "GoogleNewsRSS" if feedparser else "RSS_DISABLED(feedparser_missing)"],
    }


# ==========================
# 9) Auth APIs
# ==========================
@app.post("/register")
async def register(user: UserCreate):
    u = (user.username or "").strip()
    p = (user.password or "").strip()

    # ✅ 註冊基本規則（你要求的）
    if not USERNAME_RE.match(u):
        raise HTTPException(status_code=400, detail="帳號規則：4–20碼，只允許英文/數字/底線")
    if not PASSWORD_RE.match(p):
        raise HTTPException(status_code=400, detail="密碼規則：至少8碼，需包含英文 + 數字")

    conn = get_db()
    c = conn.cursor()

    try:
        hashed = get_password_hash(p)
        now = datetime.datetime.utcnow().isoformat()
        c.execute(
            "INSERT INTO users (username, hashed_password, created_at) VALUES (?, ?, ?)",
            (u, hashed, now),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(status_code=409, detail="此帳號已被註冊")
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"註冊失敗：{str(e)}")

    conn.close()
    return {
        "message": "註冊成功",
        "store": {
            "database": DATABASE_PATH,
            "table": "users",
            "fields": ["username", "hashed_password", "created_at"],
            "note": "密碼不會明文保存，會以 bcrypt 雜湊保存。",
        },
    }


@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db()
    c = conn.cursor()

    c.execute("SELECT * FROM users WHERE username = ?", (form_data.username,))
    row = c.fetchone()
    conn.close()

    if not row or not verify_password(form_data.password, row["hashed_password"]):
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")

    token = create_access_token(subject=row["username"])
    return {"access_token": token, "token_type": "bearer"}


# ==========================
# 10) News API（真實新聞可點擊，最新優先）
# ==========================
@app.get("/api/news")
async def api_news(q: str = Query("全球市場 財經", description="可傳 symbol 或關鍵字"),
                   limit: int = Query(10, ge=1, le=20)):
    items = fetch_real_news(q, limit=limit)
    if not items:
        # 讓前端看到原因（例如缺 feedparser）
        if feedparser is None:
            raise HTTPException(status_code=500, detail="RSS 模組未啟用：缺少 feedparser 套件（下一步會在 requirements.txt 補上）")
        return []
    return items


# ==========================
# 11) Analyze API（核心）
# ==========================
@app.post("/api/analyze")
async def analyze_stock(request: AnalysisRequest):
    t0 = time.time()
    symbol = (request.symbol or "").strip()

    if not symbol:
        raise HTTPException(status_code=400, detail="symbol 不可為空")

    # 1) 抓行情
    df = fetch_price_history(symbol, period="1y")
    if df.empty:
        raise HTTPException(status_code=404, detail="找不到此股票或沒有公開行情資料")

    if len(df) < 80:
        raise HTTPException(status_code=400, detail="資料筆數不足（至少需要約80個交易日）")

    current_price = float(df["Close"].iloc[-1])
    last_date = df.index[-1].strftime("%Y-%m-%d")

    # 2) 四大面向評分（deterministic）
    tech = score_technical(df)
    fund = score_fundamental(symbol)
    chip = score_chip_proxy(df)

    # 3) 新聞（用 symbol 搜尋，最新優先）
    news_items = fetch_real_news(symbol, limit=10)
    news_score = score_news_sentiment(news_items)

    overall = composite_score(tech["score"], fund["score"], chip["score"], news_score["score"])
    senti = sentiment_text(overall)

    # 4) 資金配置試算
    principal = float(request.principal or 0)
    if principal <= 0:
        raise HTTPException(status_code=400, detail="本金 principal 必須大於 0")

    max_shares = int(principal // current_price)
    total_cost = float(max_shares * current_price)
    remain_cash = float(principal - total_cost)
    risk_loss_10 = float(total_cost * 0.10)

    # 5) 波段價位
    trade = band_trade_prices(current_price)

    # 6) ROI 預估
    roi = estimate_roi(total_cost, df)

    # 7) 極端行情預警（60天）
    risk = extreme_risk_95(total_cost, df, horizon_days=60)

    # 8) 圖表資料（history + 預測區間帶）
    chart = build_chart_data(df, future_days=60)

    elapsed = round(time.time() - t0, 3)

    return {
        "symbol": symbol.upper(),
        "data_status": {
            "price_rows": int(len(df)),
            "price_last_date": last_date,
            "fetch_seconds": elapsed,
            "sources": ["yfinance", "GoogleNewsRSS" if feedparser else "RSS_DISABLED(feedparser_missing)"],
        },
        "price": round(current_price, 2),

        "ai_score": overall,
        "ai_sentiment": senti,

        "score_breakdown": {
            "technical": tech,
            "fundamental": fund,
            "chip": chip,
            "news": {**news_score, "news_count": len(news_items)},
        },

        "money_management": {
            "principal": int(round(principal)),
            "max_shares": max_shares,
            "total_cost": int(round(total_cost)),
            "remain_cash": int(round(remain_cash)),
            "risk_loss_10_percent": int(round(risk_loss_10)),
        },

        "advice": trade,

        "roi_estimates": roi,

        "risk_analysis": risk,

        "chart_data": chart,

        # ✅ 讓前端顯示「最新新聞」但你之前說「個股新聞與AI觀點可刪」
        # 這裡不強制回新聞內容；若你要在前端顯示可開啟：
        # "news_items": news_items,
    }


# ==========================
# 12) Favorites（需登入）
# ==========================
@app.get("/api/favorites")
async def get_favorites(user: str = Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT symbol, created_at FROM favorites WHERE username=? ORDER BY created_at DESC", (user,))
    rows = c.fetchall()
    conn.close()
    return [{"symbol": r["symbol"], "created_at": r["created_at"]} for r in rows]


@app.post("/api/favorites")
async def add_favorite(req: FavoriteReq, user: str = Depends(get_current_user)):
    sym = (req.symbol or "").strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol 不可為空")

    conn = get_db()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT OR IGNORE INTO favorites (username, symbol, created_at) VALUES (?, ?, ?)",
            (user, sym, datetime.datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return {"message": "ok", "symbol": sym}


@app.delete("/api/favorites")
async def remove_favorite(symbol: str = Query(...), user: str = Depends(get_current_user)):
    sym = (symbol or "").strip().upper()
    conn = get_db()
    c = conn.cursor()
    c.execute("DELETE FROM favorites WHERE username=? AND symbol=?", (user, sym))
    conn.commit()
    conn.close()
    return {"message": "ok", "symbol": sym}


# ==========================
# 13) Portfolio（需登入）
# ==========================
@app.get("/api/portfolio")
async def get_portfolio(user: str = Depends(get_current_user)):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT symbol, shares, avg_cost, created_at FROM portfolio WHERE username=? ORDER BY created_at DESC", (user,))
    rows = c.fetchall()
    conn.close()

    holdings = []
    total_cost = 0.0
    total_value = 0.0

    # 為了「真實資料」，這裡會抓現價（可能慢，但是真）
    for r in rows:
        sym = r["symbol"]
        shares = int(r["shares"])
        avg_cost = float(r["avg_cost"])
        df = fetch_price_history(sym, period="3mo")
        cur = float(df["Close"].iloc[-1]) if not df.empty else avg_cost
        mv = shares * cur
        cost = shares * avg_cost

        holdings.append({
            "symbol": sym,
            "shares": shares,
            "avg_cost": round(avg_cost, 2),
            "current_price": round(cur, 2),
            "market_value": int(round(mv)),
            "pnl": int(round(mv - cost)),
            "created_at": r["created_at"],
        })
        total_cost += cost
        total_value += mv

    return {
        "total_asset": int(round(total_value)),
        "total_cost": int(round(total_cost)),
        "unrealized_pnl": int(round(total_value - total_cost)),
        "holdings": holdings,
    }


@app.post("/api/portfolio/add")
async def add_to_portfolio(item: PortfolioItem, user: str = Depends(get_current_user)):
    sym = (item.symbol or "").strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol 不可為空")
    if item.shares <= 0:
        raise HTTPException(status_code=400, detail="shares 必須 > 0")
    if item.cost <= 0:
        raise HTTPException(status_code=400, detail="cost 必須 > 0")

    conn = get_db()
    c = conn.cursor()
    c.execute(
        "INSERT INTO portfolio (username, symbol, shares, avg_cost, created_at) VALUES (?, ?, ?, ?, ?)",
        (user, sym, int(item.shares), float(item.cost), datetime.datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()
    return {"message": "Added to portfolio", "symbol": sym}


# ==========================
# 14) K 線詳細分析（需登入，先提供基礎版本）
# ==========================
def detect_candle_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    提供部分常見型態偵測（不假裝全48型態都精準判定，先打底）
    後續可逐步擴充到 48 型態。
    """
    if len(df) < 5:
        return {"patterns": [], "note": "資料不足"}

    last = df.iloc[-1]
    prev = df.iloc[-2]

    o, h, l, c = float(last["Open"]), float(last["High"]), float(last["Low"]), float(last["Close"])
    o2, c2 = float(prev["Open"]), float(prev["Close"])

    body = abs(c - o)
    rng = max(1e-9, h - l)
    upper = h - max(o, c)
    lower = min(o, c) - l

    patterns = []

    # Doji
    if body / rng < 0.1:
        patterns.append("十字線 Doji")

    # Hammer / Hanging man（簡化：下影線長、實體小）
    if lower / rng > 0.55 and body / rng < 0.25:
        patterns.append("錘子線 Hammer（需搭配趨勢判讀）")

    # Engulfing
    # 多方吞噬：前一根黑、這根紅且實體包住
    if (c2 < o2) and (c > o) and (c >= o2) and (o <= c2):
        patterns.append("多方吞噬 Bullish Engulfing")
    # 空方吞噬：前一根紅、這根黑且實體包住
    if (c2 > o2) and (c < o) and (o >= c2) and (c <= o2):
        patterns.append("空方吞噬 Bearish Engulfing")

    return {
        "patterns": patterns,
        "candle": {
            "body": round(body, 4),
            "range": round(rng, 4),
            "upper_shadow": round(upper, 4),
            "lower_shadow": round(lower, 4),
        },
        "note": "此為基礎型態偵測，後續可擴充到 48 種型態完整判別。",
    }


@app.get("/api/kline/detail")
async def kline_detail(
    symbol: str = Query(...),
    interval: str = Query("1d", description="1d/1wk/1mo"),
    lookback: int = Query(200, ge=60, le=800),
    user: str = Depends(get_current_user),
):
    """
    ✅ 需要登入才能使用（你要求的）
    回傳：
    - K 線型態（基礎）
    - 時間週期
    - 量能
    - 技術指標（MA/MACD/RSI/KD/布林）
    """
    # yfinance interval: 1d, 1wk, 1mo
    df = yf.download(symbol, period="2y", interval=interval, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="找不到此股票K線資料")

    df = df.dropna(subset=["Close"])
    df = df.tail(lookback)

    tech = score_technical(df)
    patterns = detect_candle_patterns(df)

    # 量能（近5 vs 近20）
    vol = df["Volume"].fillna(0)
    v5 = float(vol.tail(5).mean())
    v20 = float(vol.tail(20).mean())
    vol_ratio = (v5 / v20) if v20 > 0 else 1.0

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "lookback": lookback,
        "last_date": df.index[-1].strftime("%Y-%m-%d"),
        "volume": {
            "v5": int(round(v5)),
            "v20": int(round(v20)),
            "ratio_5v20": round(vol_ratio, 4),
        },
        "technical": tech,
        "patterns": patterns,
        "hint": {
            "48_patterns": "你要求的48種型態可逐步擴充；目前先上常見核心型態+指標整合。",
        },
        "access": {"user": user, "auth_required": True},
    }


# ==========================
# 15) 啟動
# ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
