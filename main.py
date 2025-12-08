import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

app = FastAPI()

# 設定 CORS (允許所有來源連線，方便開發與上架)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 資料模型 ---
class StockRequest(BaseModel):
    ticker: str
    principal: float
    risk: str

class ScreenRequest(BaseModel):
    strategy: str

# --- 擴充後的掃描清單 ---
STOCK_POOLS = {
    "growth": ["2330.TW", "2454.TW", "2317.TW", "2382.TW", "2308.TW", "NVDA", "AMD", "TSLA", "MSFT", "AAPL", "PLTR", "AVGO", "META"],
    "dividend": ["0050.TW", "0056.TW", "00878.TW", "2412.TW", "2891.TW", "KO", "JNJ", "PG", "VZ", "T", "O"],
    "value": ["2303.TW", "2002.TW", "1301.TW", "1101.TW", "2603.TW", "INTC", "PYPL", "DIS", "F", "GM"],
    "decline": ["TSLA", "INTC", "NKE", "SBUX", "1301.TW", "2603.TW", "2409.TW"]
}

# --- 輔助函式：單股深度分析 ---
def calculate_depth_analysis(ticker, principal, risk):
    try:
        ticker = ticker.upper().strip()
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        
        if hist.empty:
            if not ticker.endswith(".TW") and ticker.isdigit():
                ticker += ".TW"
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
            if hist.empty:
                return {"error": "找不到股票數據"}

        current_price = hist['Close'].iloc[-1]
        
        # 技術指標計算
        hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
        hist['SMA_60'] = hist['Close'].rolling(window=60).mean()
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        loss = loss.replace(0, 0.001)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

        # 評分
        tech_score = 50
        if current_price > hist['SMA_20'].iloc[-1]: tech_score += 15
        if hist['SMA_20'].iloc[-1] > hist['SMA_60'].iloc[-1]: tech_score += 15
        if 30 < current_rsi < 70: tech_score += 10
        elif current_rsi <= 30: tech_score += 20
        
        fund_score = 60 # 簡化
        chip_score = np.random.randint(40, 80)
        news_score = np.random.randint(40, 80)
        total_score = int(tech_score * 0.35 + fund_score * 0.3 + chip_score * 0.2 + news_score * 0.15)
        
        evaluation = "觀望"
        if total_score >= 70: evaluation = "偏多 (買進)"
        elif total_score <= 40: evaluation = "偏空 (賣出)"

        # 預測
        hist['Returns'] = hist['Close'].pct_change()
        mu = hist['Returns'].mean()
        sigma = hist['Returns'].std()
        
        days_predict = 60
        future_dates = [(datetime.now() + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(1, days_predict + 1)]
        
        proj_mean, proj_upper, proj_lower = [], [], []
        last_p = current_price
        
        for i in range(1, days_predict + 1):
            drift = (mu - 0.5 * sigma**2) * i
            shock = sigma * np.sqrt(i)
            proj_mean.append(last_p * np.exp(drift))
            proj_upper.append(last_p * np.exp(drift + shock))
            proj_lower.append(last_p * np.exp(drift - shock))

        # ROI
        roi_data = {}
        periods = {"short": 5, "mid": 60, "long": 250}
        for k, d in periods.items():
            r_fac = 0.5 if risk == "aggressive" else (-0.2 if risk == "conservative" else 0)
            exp_ret = (mu * d) + (sigma * np.sqrt(d) * r_fac)
            prof = principal * exp_ret
            roi_data[k] = {
                "return_pct": round(exp_ret * 100, 2),
                "profit_cash": int(prof),
                "final_amount": int(principal + prof)
            }

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "total_score": total_score,
            "evaluation": evaluation,
            "recommendation": "長期持有" if total_score > 60 else "短期觀望",
            "roi": roi_data,
            "details": {"tech": int(tech_score), "fund": int(fund_score), "chip": chip_score, "news": news_score},
            "chart_data": {
                "history_date": hist.index.strftime('%Y-%m-%d').tolist(),
                "history_price": hist['Close'].tolist(),
                "future_date": future_dates,
                "future_mean": proj_mean,
                "future_upper": proj_upper,
                "future_lower": proj_lower
            }
        }
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

# --- 輔助函式：快速掃描 ---
def quick_scan_stock(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        if hist.empty: return None
        
        price = hist['Close'].iloc[-1]
        prev = hist['Close'].iloc[0]
        change = ((price - prev) / prev) * 100
        
        score = 50 + (change if change > 0 else change * 0.5)
        return {
            "ticker": ticker,
            "price": round(price, 2),
            "change_pct": round(change, 2),
            "score": int(max(0, min(100, score)))
        }
    except:
        return None

# --- API 路由 ---
@app.get("/")
def home():
    return {"message": "AI Stock Hunter Backend is Running!"}

@app.post("/analyze")
async def api_analyze(req: StockRequest):
    return calculate_depth_analysis(req.ticker, req.principal, req.risk)

@app.post("/screen")
async def api_screen(req: ScreenRequest):
    target_list = STOCK_POOLS.get(req.strategy, [])
    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(quick_scan_stock, t) for t in target_list]
        for f in futures:
            if r := f.result(): results.append(r)
    
    results.sort(key=lambda x: x['score'], reverse=True)
    return {"results": results[:10]}

if __name__ == "__main__":
    # 雲端部署關鍵修改：
    # 1. host 必須是 0.0.0.0 (表示允許外部連線)
    # 2. port 必須讀取環境變數，否則雲端會報錯
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)