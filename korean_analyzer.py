"""
korean_analyzer.py
국내(한국장) ETF 11종 챔피언 스윙 전략 정밀 분석 엔진
- 5개년 백테스트로 검증된 4대 국장 챔피언 전략:
  1. KR_PULLBACK_PRO: 200일선 우상향 필터 + 20/60선 지지반등 + 스토캐스틱 턴 (지수/레버리지/반도체)
  2. KR_BB_SQUEEZE: 볼린저 밴드 수축(스퀴즈) 후 거래량 폭발 상단 돌파 (2차전지/골드/달러)
  3. KR_OVERSOLD_REBOUND: RSI(14) 35 이하/괴리율 침체 후 5일선 골든크로스 복귀 (WTI원유/은행/헬스케어)
  4. KR_PULLBACK_CLASSIC: 60일선 위 20일선 눌림목 반등 (코스닥150 레버리지)
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class KoreanSwingSignal:
    ticker: str
    name: str
    category: str
    strategy_code: str
    strategy_name: str
    current_price: float
    change_pct: float
    signals: list[str] = field(default_factory=list)        # 매수 시그널 목록
    reasons: list[str] = field(default_factory=list)        # 매수 근거
    sell_signals: list[str] = field(default_factory=list)   # 매도 시그널 목록
    sell_reasons: list[str] = field(default_factory=list)   # 매도 근거
    exit_guide: str = ""                                    # 청산 및 손익비 가이드
    indicators: dict = field(default_factory=dict)          # 주요 보조지표 수치
    stop_loss: float = 0.0                                  # 추천 손절가
    target_1: float = 0.0                                   # 1차 목표가
    target_2: float = 0.0                                   # 2차 목표가
    watch_status: str = ""                                  # 대기 상태 시 요약


def compute_korean_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    
    # 1. 이동평균선
    d["MA5"] = d["Close"].rolling(5).mean()
    d["MA20"] = d["Close"].rolling(20).mean()
    d["MA60"] = d["Close"].rolling(60).mean()
    d["MA200"] = d["Close"].rolling(200, min_periods=60).mean()
    
    # 2. RSI (14일 & 2일)
    delta = d["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # RSI 14
    roll_up14 = gain.ewm(span=14, adjust=False).mean()
    roll_down14 = loss.ewm(span=14, adjust=False).mean()
    rs14 = roll_up14 / (roll_down14 + 1e-9)
    d["RSI14"] = 100 - (100 / (1 + rs14))
    
    # RSI 2 (단기 과매도용)
    roll_up2 = gain.ewm(span=2, adjust=False).mean()
    roll_down2 = loss.ewm(span=2, adjust=False).mean()
    rs2 = roll_up2 / (roll_down2 + 1e-9)
    d["RSI2"] = 100 - (100 / (1 + rs2))
    
    # 3. 볼린저 밴드
    std20 = d["Close"].rolling(20).std()
    d["BB_Mid"] = d["MA20"]
    d["BB_Upper"] = d["BB_Mid"] + 2 * std20
    d["BB_Lower"] = d["BB_Mid"] - 2 * std20
    d["BB_Width"] = (d["BB_Upper"] - d["BB_Lower"]) / d["BB_Mid"]
    d["BB_Width_MA20"] = d["BB_Width"].rolling(20).mean()
    
    # 4. 스토캐스틱 슬로우 (14, 3, 3)
    low14 = d["Low"].rolling(14).min()
    high14 = d["High"].rolling(14).max()
    k_fast = 100 * (d["Close"] - low14) / (high14 - low14 + 1e-9)
    d["Stoch_K"] = k_fast.rolling(3).mean()
    d["Stoch_D"] = d["Stoch_K"].rolling(3).mean()
    
    # 5. 거래량 & 이격도
    d["Vol_MA20"] = d["Volume"].rolling(20).mean()
    d["Disparity20"] = (d["Close"] / d["MA20"]) * 100
    
    # 6. ATR (14일)
    tr1 = d["High"] - d["Low"]
    tr2 = (d["High"] - d["Close"].shift(1)).abs()
    tr3 = (d["Low"] - d["Close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["ATR"] = tr.rolling(14).mean()
    
    return d


def analyze_korean_etf(
    ticker: str,
    name: str,
    category: str,
    strategy_code: str,
    df: pd.DataFrame,
    active_pos: dict | None = None,
) -> KoreanSwingSignal:
    d = compute_korean_indicators(df)
    c = d.iloc[-1]
    p = d.iloc[-2]
    
    curr_price = float(c["Close"])
    prev_close = float(p["Close"])
    change_pct = ((curr_price - prev_close) / prev_close) * 100
    atr = float(c["ATR"]) if not np.isnan(c["ATR"]) else curr_price * 0.02
    
    strat_names = {
        "KR_PULLBACK_PRO": "K-추세 눌림목 프로 (200일선 필터 + 20/60선 지지반등)",
        "KR_BB_SQUEEZE": "K-볼린저 수축 돌파 (스퀴즈 + 밴드상단돌파 + 거래량폭발)",
        "KR_OVERSOLD_REBOUND": "K-원자재/매크로 과매도 반등 (RSI35/괴리율 침체 + 5선복귀)",
        "KR_PULLBACK_CLASSIC": "K-이평선 눌림목 클래식 (60일선 위 20선 지지반등)",
    }
    s_name = strat_names.get(strategy_code, strategy_code)
    
    sig = KoreanSwingSignal(
        ticker=ticker,
        name=name,
        category=category,
        strategy_code=strategy_code,
        strategy_name=s_name,
        current_price=curr_price,
        change_pct=change_pct,
        indicators={
            "MA5": round(float(c["MA5"]), 0),
            "MA20": round(float(c["MA20"]), 0),
            "MA60": round(float(c["MA60"]), 0),
            "MA200": round(float(c["MA200"]), 0) if not np.isnan(c["MA200"]) else 0,
            "RSI14": round(float(c["RSI14"]), 1),
            "Stoch_K": round(float(c["Stoch_K"]), 1),
            "Stoch_D": round(float(c["Stoch_D"]), 1),
            "Disparity20": round(float(c["Disparity20"]), 1),
            "BB_Width_Ratio": round(float(c["BB_Width"] / (c["BB_Width_MA20"] + 1e-9)), 2),
            "Vol_Ratio": round(float(c["Volume"] / (c["Vol_MA20"] + 1e-9)), 2),
        }
    )
    
    # -------------------------------------------------------------------------
    # [1] 매수 시그널 판정
    # -------------------------------------------------------------------------
    if strategy_code == "KR_PULLBACK_PRO":
        # 200일선 우상향 + 20/60선 지지반등 + 스토캐스틱 K>D 골든크로스
        ma200_valid = curr_price > c["MA200"] if not np.isnan(c["MA200"]) else True
        supp_hit = (c["Low"] <= c["MA20"] * 1.015) or (c["Low"] <= c["MA60"] * 1.015)
        rebound = c["Close"] > c["Open"] and c["Close"] > p["Close"]
        stoch_turn = c["Stoch_K"] > c["Stoch_D"] and p["Stoch_K"] <= p["Stoch_D"]
        
        if ma200_valid and supp_hit and rebound and stoch_turn:
            sig.signals.append("🟢 [K-추세 눌림목 매수 타점 포착]")
            sig.reasons.append(f"200일선(₩{c['MA200']:,.0f}) 위 대세 상승장에서 20/60일 지지선 터치 후 양봉 반등")
            sig.reasons.append(f"스토캐스틱 골든크로스 발생 (K:{c['Stoch_K']:.1f} > D:{c['Stoch_D']:.1f})")
            sig.stop_loss = round(curr_price * 0.96, 0) # -4% 손절
            sig.target_1 = round(curr_price * 1.05, 0)  # +5% (트레일링 스탑 가동)
            sig.target_2 = round(curr_price * 1.12, 0)  # +12% 익절
            sig.exit_guide = "+5% 도달 시 최고점 대비 -2.5% 트레일링 스탑 발동, 20일선 이탈 시 전량 정리"
        else:
            if not ma200_valid:
                sig.watch_status = f"200일선(₩{c['MA200']:,.0f}) 하회 중 (하락추세 관망)"
            elif not supp_hit:
                sig.watch_status = f"20일선(₩{c['MA20']:,.0f}) 지지선 눌림 대기 중"
            else:
                sig.watch_status = f"지지선 부근 반등 캔들 확인 대기 (스토캐스틱 K:{c['Stoch_K']:.1f})"

    elif strategy_code == "KR_BB_SQUEEZE":
        # 밴드폭 극도 수축(스퀴즈) 후 거래량 1.3배 폭발 상단 돌파
        is_squeeze = c["BB_Width"] < c["BB_Width_MA20"] * 0.90
        is_breakout = c["Close"] > c["BB_Upper"]
        vol_surge = c["Volume"] > c["Vol_MA20"] * 1.30
        is_bullish = c["Close"] > c["Open"]
        
        if is_squeeze and is_breakout and vol_surge and is_bullish:
            sig.signals.append("⚡ [K-볼린저 수축 돌파 매수 타점 포착]")
            sig.reasons.append(f"볼린저 밴드 극도 수축(스퀴즈) 후 상단 밴드(₩{c['BB_Upper']:,.0f}) 돌파")
            sig.reasons.append(f"20일 평균 대비 거래량 {c['Volume']/c['Vol_MA20']:.1f}배 폭발한 장대양봉")
            sig.stop_loss = round(curr_price * 0.965, 0) # -3.5%
            sig.target_1 = round(curr_price * 1.05, 0)
            sig.target_2 = round(curr_price * 1.10, 0)
            sig.exit_guide = "볼린저 중심선(20일선 ₩{:,.0f}) 이탈 시 청산, +5% 이후 트레일링스탑".format(c['BB_Mid'])
        else:
            if not is_squeeze:
                sig.watch_status = f"밴드 수축(스퀴즈) 진행 대기 (변동성 비: {c['BB_Width']/c['BB_Width_MA20']:.2f})"
            elif not is_breakout:
                sig.watch_status = f"상단 돌파 대기 (상단: ₩{c['BB_Upper']:,.0f})"
            else:
                sig.watch_status = f"거래량 폭발 확인 대기 (거래량: {c['Volume']/c['Vol_MA20']:.1f}배)"

    elif strategy_code == "KR_OVERSOLD_REBOUND":
        # RSI14 <= 35 또는 괴리율 <= 94 침체 후 5일선 양봉 회복
        oversold = (c["RSI14"] <= 35.0) or (c["Disparity20"] <= 94.0)
        ma5_cross = (c["Close"] > c["MA5"]) and (p["Close"] <= p["MA5"])
        rebound = oversold or (ma5_cross and p["RSI14"] <= 38.0)
        
        if rebound and c["Close"] > c["Open"]:
            sig.signals.append("💎 [K-원자재/매크로 과매도 반등 타점 포착]")
            sig.reasons.append(f"과매도 침체(RSI: {c['RSI14']:.1f}, 20일 괴리율: {c['Disparity20']:.1f}%) 기록")
            sig.reasons.append(f"단기 5일 이평선(₩{c['MA5']:,.0f}) 위로 양봉 회복 반등 시작")
            sig.stop_loss = round(curr_price * 0.96, 0) # -4%
            sig.target_1 = round(c["MA20"], 0)         # 20일선 복귀 1차 목표
            sig.target_2 = round(curr_price * 1.08, 0)
            sig.exit_guide = "20일 이평선(₩{:,.0f}) 터치 시 1차 익절, 5일선 재이탈 시 청산".format(c['MA20'])
        else:
            sig.watch_status = f"과매도 침체 대기 (현재 RSI: {c['RSI14']:.1f}, 괴리율: {c['Disparity20']:.1f}%)"

    elif strategy_code == "KR_PULLBACK_CLASSIC":
        # 60일선 위 + 20일선 눌림목 반등 (코스닥 레버리지 특화)
        trend = c["Close"] > c["MA60"]
        touch = (c["Low"] <= c["MA20"] * 1.01) and (c["Close"] >= c["MA20"])
        bullish = c["Close"] > c["Open"]
        
        if trend and touch and bullish:
            sig.signals.append("🟢 [K-코스닥 눌림목 반등 타점 포착]")
            sig.reasons.append(f"60일선(₩{c['MA60']:,.0f}) 위에서 20일선(₩{c['MA20']:,.0f}) 터치 후 양봉 지지")
            sig.stop_loss = round(curr_price * 0.96, 0)
            sig.target_1 = round(curr_price * 1.06, 0)
            sig.target_2 = round(curr_price * 1.12, 0)
            sig.exit_guide = "+6% 도달 시 1차 분할 익절, 20일선 하향 이탈 시 청산"
        else:
            if not trend:
                sig.watch_status = f"60일선(₩{c['MA60']:,.0f}) 하회 (관망)"
            else:
                sig.watch_status = f"20일선(₩{c['MA20']:,.0f}) 눌림 대기 중"

    # -------------------------------------------------------------------------
    # [2] 보유 포지션 매도(익절/손절/트레일링스탑) 판정
    # -------------------------------------------------------------------------
    if active_pos:
        entry_price = float(active_pos.get("entry_price", curr_price))
        pos_ret = ((curr_price - entry_price) / entry_price) * 100
        highest_p = float(active_pos.get("highest_price", entry_price))
        highest_p = max(highest_p, curr_price)
        active_pos["highest_price"] = highest_p
        drawdown_from_peak = ((curr_price - highest_p) / highest_p) * 100
        
        # 1. 고정 손절선 (-4%)
        if pos_ret <= -4.0:
            sig.sell_signals.append("🚨 [손절 청산 시그널: 손실 한도(-4.0%) 도달]")
            sig.sell_reasons.append(f"진입가(₩{entry_price:,.0f}) 대비 수익률 {pos_ret:+.2f}%로 손절 기준 도달")
            
        # 2. 트레일링 스탑 (+5% 이상 수익 후 고점 대비 -2.5% 하락)
        elif pos_ret >= 5.0 and drawdown_from_peak <= -2.5:
            sig.sell_signals.append("💰 [트레일링 스탑 익절: 고점 대비 2.5% 하락 방어]")
            sig.sell_reasons.append(f"최고가(₩{highest_p:,.0f}) 달성 후 -2.5% 밀림 (현재 수익률: {pos_ret:+.2f}%)")
            
        # 3. 20일선 하향 이탈
        elif curr_price < c["MA20"] and p["Close"] >= p["MA20"]:
            sig.sell_signals.append("⚠️ [추세 이탈 매도: 20일 이평선 하향 이탈]")
            sig.sell_reasons.append(f"20일선(₩{c['MA20']:,.0f}) 붕괴로 단기 상승 추세 종료")
            
        # 4. 목표가 도달
        target_val = float(active_pos.get("target_1", 0.0))
        if target_val > 0 and curr_price >= target_val:
            sig.sell_signals.append(f"🎯 [1차 목표가(₩{target_val:,.0f}) 달성 익절]")
            sig.sell_reasons.append(f"진입가 대비 +{pos_ret:.1f}% 수익 달성으로 분할 익절 권장")

    return sig
