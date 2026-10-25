"""
미국 ETF 스윙 시그널 봇 - 기술적 지표 계산 및 3대 스윙 전략 판별 엔진
"""
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from config import STRATEGY_PARAMS


@dataclass
class SwingSignalResult:
    ticker: str
    name: str
    category: str
    current_price: float
    change_pct: float
    signals: list[str] = field(default_factory=list)       # 포착된 매수 전략 태그 목록
    reasons: list[str] = field(default_factory=list)       # 상세 매수 차트 근거 목록
    sell_signals: list[str] = field(default_factory=list)  # 포착된 매도(익절/손절) 시그널 목록
    sell_reasons: list[str] = field(default_factory=list)  # 상세 매도 차트 근거 목록
    exit_guide: str = ""                                   # 전략별 맞춤 청산(익절) 가이드
    indicators: dict = field(default_factory=dict)         # 주요 지표 요약
    stop_loss: float = 0.0                                 # 추천 손절가 (ATR 기반)
    target_1: float = 0.0                                  # 1차 익절 목표가
    target_2: float = 0.0                                  # 2차 익절 목표가
    watch_status: str = ""                                 # 시그널 미발생 시 현재 상태 요약


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI(상대강도지수) 계산"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """스윙 매매에 필요한 핵심 기술적 지표를 계산합니다."""
    d = df.copy()

    # 1. 이동평균선 (5일, 20일, 50일, 200일)
    d["SMA5"] = d["Close"].rolling(window=5).mean()
    d["SMA20"] = d["Close"].rolling(window=20).mean()
    d["SMA50"] = d["Close"].rolling(window=50).mean()
    d["SMA200"] = d["Close"].rolling(window=200, min_periods=100).mean()

    # 2. RSI (14일 표준 & 2일 단기 과매도용)
    d["RSI14"] = calculate_rsi(d["Close"], 14)
    d["RSI2"] = calculate_rsi(d["Close"], 2)

    # 3. 볼린저 밴드 (20일, 2표준편차)
    std20 = d["Close"].rolling(window=20).std()
    d["BB_Upper"] = d["SMA20"] + (std20 * 2)
    d["BB_Lower"] = d["SMA20"] - (std20 * 2)
    d["BB_Width"] = (d["BB_Upper"] - d["BB_Lower"]) / d["SMA20"] * 100

    # 4. MACD (12, 26, 9)
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["MACD"] = ema12 - ema26
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()
    d["MACD_Hist"] = d["MACD"] - d["MACD_Signal"]

    # 5. 거래량 이동평균 (20일) 및 거래량 비율
    d["Vol_SMA20"] = d["Volume"].rolling(window=20).mean()
    d["Vol_Ratio"] = d["Volume"] / d["Vol_SMA20"].replace(0, np.nan)

    # 6. ATR (Average True Range, 14일) - 변동성 기반 손절/익절가 산출용
    high_low = d["High"] - d["Low"]
    high_close = (d["High"] - d["Close"].shift(1)).abs()
    low_close = (d["Low"] - d["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d["ATR14"] = tr.rolling(window=14).mean()

    # 7. 최근 N일 최고가 (당일 제외 전일까지의 N일 고점 - 돌파 판별용)
    lookback = STRATEGY_PARAMS["breakout_lookback_days"]
    d["Prev_High_N"] = d["High"].shift(1).rolling(window=lookback).max()
    d["Prev_Close_High_N"] = d["Close"].shift(1).rolling(window=lookback).max()
    d["Recent_High_10"] = d["High"].rolling(window=10).max()

    # 8. 이동평균선 기울기 (최근 5거래일 전 대비 변화율 %)
    d["SMA20_Slope"] = (d["SMA20"] - d["SMA20"].shift(5)) / d["SMA20"].shift(5) * 100.0
    d["SMA50_Slope"] = (d["SMA50"] - d["SMA50"].shift(5)) / d["SMA50"].shift(5) * 100.0
    d["SMA200_Slope"] = (d["SMA200"] - d["SMA200"].shift(10)) / d["SMA200"].shift(10) * 100.0

    return d


def analyze_etf_chart(
    ticker: str,
    name: str,
    category: str,
    df: pd.DataFrame,
    active_pos: dict | None = None,
    strategy_mode: str = "OVERSOLD_ONLY",
) -> SwingSignalResult:
    """
    계산된 기술적 지표를 바탕으로 종목별 최적화된 스마트 라우팅(strategy_mode) 시그널을 판별합니다.
      - OVERSOLD_ONLY: 주가지수/반도체/3배레버리지 -> 승률 80%+ [💎과매도 반등] 전용
      - MACRO_CYCLE  : 에너지/금/원자재/채권 -> [🔥거래량 폭발 박스권 돌파] + [💎과매도 반등] 병행
    """
    d = compute_indicators(df)
    curr = d.iloc[-1]
    prev = d.iloc[-2]

    close = float(curr["Close"])
    open_p = float(curr["Open"])
    prev_close = float(prev["Close"])
    change_pct = ((close - prev_close) / prev_close) * 100.0

    sma5 = float(curr["SMA5"])
    sma20 = float(curr["SMA20"])
    prev_sma20 = float(prev["SMA20"])
    sma50 = float(curr["SMA50"])
    sma200 = float(curr["SMA200"]) if not pd.isna(curr["SMA200"]) else sma50
    sma20_slope = float(curr["SMA20_Slope"]) if not pd.isna(curr["SMA20_Slope"]) else 0.0
    sma50_slope = float(curr["SMA50_Slope"]) if not pd.isna(curr["SMA50_Slope"]) else 0.0
    sma200_slope = float(curr["SMA200_Slope"]) if not pd.isna(curr["SMA200_Slope"]) else 0.0

    rsi14 = float(curr["RSI14"])
    prev_rsi14 = float(prev["RSI14"])
    rsi2 = float(curr["RSI2"])
    prev_rsi2 = float(prev["RSI2"])
    vol_ratio = float(curr["Vol_Ratio"]) if not pd.isna(curr["Vol_Ratio"]) else 1.0
    atr = float(curr["ATR14"]) if not pd.isna(curr["ATR14"]) else close * 0.02
    bb_upper = float(curr["BB_Upper"])
    bb_lower = float(curr["BB_Lower"])
    macd_hist = float(curr["MACD_Hist"])
    prev_macd_hist = float(prev["MACD_Hist"])
    recent_high_10 = float(curr["Recent_High_10"])
    prev_high_close = float(curr["Prev_Close_High_N"])

    # 이평선 이격도 (%)
    dist_sma20_pct = ((close - sma20) / sma20) * 100.0
    dist_sma50_pct = ((close - sma50) / sma50) * 100.0

    signals = []
    reasons = []
    sell_signals = []
    sell_reasons = []

    # =========================================================================
    # [A] 매수(진입) 시그널 판별 — 종목별 스마트 라우팅(Smart Strategy Routing)
    # =========================================================================

    # 1) 💎 과매도 역추세 반등 매매 (평균 보유 3~4일, 승률 ~80% 주력 스나이퍼)
    rsi14_thresh = STRATEGY_PARAMS["oversold_rsi14_threshold"]
    rsi2_thresh = STRATEGY_PARAMS["oversold_rsi2_threshold"]

    cond_oversold_standard = (
        (rsi14 <= rsi14_thresh or prev_rsi14 <= rsi14_thresh)
        and (close <= bb_lower * 1.015)
        and (close > open_p or rsi14 > prev_rsi14)
    )
    # MACRO_CYCLE(에너지/원자재)은 변동성 사이클이 길어 RSI(2)<=10 극단적 과매도일 때만 진입하고,
    # OVERSOLD_ONLY(지수/반도체/레버리지)는 RSI(2)<=15 표준 스나이핑 적용
    eff_rsi2_thresh = 10.0 if strategy_mode == "MACRO_CYCLE" else rsi2_thresh
    cond_oversold_rsi2 = (rsi2 <= eff_rsi2_thresh) and (close > sma200 * 0.96) and (sma200_slope >= -0.5)

    if cond_oversold_standard or cond_oversold_rsi2:
        signals.append("💎 [매수-과매도반등: 단기 급락 스나이핑]")
        if cond_oversold_standard:
            reasons.append(
                f"RSI(14) 과매도({rsi14:.1f}) 및 볼린저밴드 하단(${bb_lower:.2f}) 지지 반등 타점"
            )
        else:
            reasons.append(
                f"장기 추세선 위 단기 과대낙폭 발생 (RSI(2)={rsi2:.1f} 극단적 과매도) — 고승률(80%+) 평균회귀 타점"
            )

    # 2) 🔥 박스권 상단 돌파 매매 — MACRO_CYCLE(에너지/금/채권/원자재) 전용!
    #    (지수/반도체/3배레버리지에서는 가짜 돌파 손실을 막기 위해 자동 비활성화)
    if strategy_mode == "MACRO_CYCLE":
        vol_thresh = STRATEGY_PARAMS["breakout_volume_ratio"]
        candle_range = max(float(curr["High"]) - float(curr["Low"]), 0.0001)
        close_strength = (close - float(curr["Low"])) / candle_range

        is_price_breakout = (close > prev_high_close) and (prev_close <= prev_high_close * 1.002)
        is_trend_aligned = (close > sma50) and (sma20_slope > 0) and (50.0 <= rsi14 < 70.0)
        is_strong_breakout_candle = (close > open_p) and (close_strength >= 0.6)
        is_volume_and_momentum = (vol_ratio >= vol_thresh) and (macd_hist > 0) and (macd_hist > prev_macd_hist)

        if is_price_breakout and is_trend_aligned and is_strong_breakout_candle and is_volume_and_momentum:
            signals.append("🔥 [매수-박스권돌파: 거래량 폭발 신고가 돌파]")
            reasons.append(
                f"최근 {STRATEGY_PARAMS['breakout_lookback_days']}일 고점(${prev_high_close:.2f}) 돌파 + "
                f"평균 대비 거래량 {vol_ratio * 100:.0f}% 폭발 + 장대양봉 마감"
            )

    # =========================================================================
    # [B] 매도(익절/손절/청산) 시그널 판별
    # =========================================================================
    # 1) 보유 포지션(positions.json) 기반 정밀 매도 시그널
    if active_pos:
        entry_p = float(active_pos.get("entry_price", close))
        pos_strat = active_pos.get("strategy", "")
        pos_stop = float(active_pos.get("stop_loss", entry_p - atr * 2.2))
        pos_target1 = float(active_pos.get("target_1", entry_p + atr * 2.5))
        high_since = max(float(active_pos.get("highest_price", entry_p)), close, recent_high_10)
        pnl_pct = ((close - entry_p) / entry_p) * 100.0

        # 트레일링 스탑 기준가 계산
        trail_stop_p = high_since - (atr * 2.5) if high_since >= entry_p + (atr * 1.5) else pos_stop

        if close <= pos_stop:
            sell_signals.append("🚨 [매도-손절: 권장 손절선 이탈]")
            sell_reasons.append(
                f"진입가(${entry_p:.2f}) 대비 {pnl_pct:+.2f}% | 지정 손절선(${pos_stop:.2f}) 하향 이탈로 리스크 관리 청산 요망"
            )
        elif ("과매도" in pos_strat or "전략3" in pos_strat) and close > sma5 and rsi2 >= 65 and close > entry_p:
            sell_signals.append("💰 [매도-익절: 과매도 단기반등 목표 달성 (5일선 회복)]")
            sell_reasons.append(
                f"진입가(${entry_p:.2f}) 대비 {pnl_pct:+.2f}% 수익 중 | 5일 이동평균선(${sma5:.2f}) 돌파 및 RSI(2)={rsi2:.1f} 반등 완료!"
            )
        elif close >= pos_target1 and pnl_pct > 0:
            sell_signals.append("💰 [매도-익절: 1차 목표가(+2.5 ATR) 도달]")
            sell_reasons.append(
                f"진입가(${entry_p:.2f}) 대비 {pnl_pct:+.2f}% 수익 중 | 1차 목표가(${pos_target1:.2f}) 도달! 절반 익절 후 나머지는 트레일링 스탑(${trail_stop_p:.2f}) 권장"
            )
        elif high_since >= entry_p + (atr * 1.5) and close <= trail_stop_p:
            sell_signals.append("🔔 [매도-트레일링스탑: 고점 대비 -2.5 ATR 하락]")
            sell_reasons.append(
                f"최고가(${high_since:.2f}) 달성 후 트레일링 스탑 기준선(${trail_stop_p:.2f}) 이탈 — 수익 확정 청산 권장 (현재 {pnl_pct:+.2f}%)"
            )

    # 2) 차트 자체 기술적 매도 시그널 (보유 등록을 안 했더라도 차트상 명확한 매도 타점일 때 알림)
    # 조건 ①: 최근 5일 내 과매도(RSI2 <= 20)였다가 당일 5일선 위로 급반등하며 RSI2 >= 75 도달 (과매도 반등 단기 익절 타점)
    recent_min_rsi2 = float(d["RSI2"].iloc[-5:-1].min())
    if recent_min_rsi2 <= 20.0 and close > sma5 and rsi2 >= 75.0:
        if not any("과매도 단기반등" in s for s in sell_signals):
            sell_signals.append("💰 [매도-단기익절: 과매도 반등 후 5일선 회복 완료]")
            sell_reasons.append(
                f"최근 단기 과매도(RSI2 최저 {recent_min_rsi2:.1f}) 이후 5일선(${sma5:.2f}) 돌파 및 단기 과열(RSI2 {rsi2:.1f}) 도달 — 단기 스윙 차익실현 타점"
            )

    # 조건 ②: 단기 과열/목표 도달 (RSI14 >= 72 & 볼린저밴드 상단 돌파 후 음봉 또는 상승 둔화)
    if rsi14 >= 72.0 and close >= bb_upper * 0.995:
        sell_signals.append("🔥 [매도-분할익절: 단기 과열 및 볼린저밴드 상단 도달]")
        sell_reasons.append(
            f"RSI(14) 과매수({rsi14:.1f}) 및 볼린저밴드 상단(${bb_upper:.2f}) 도달 — 추세 추종 물량 일부 분할 익절 및 트레일링 스탑 권장"
        )

    # 조건 ③: 추세 이탈 경보 (20일선 위에 있던 종목이 거래량을 동반하거나 RSI < 45로 20일선을 강하게 하향 이탈한 당일)
    if (prev_close >= prev_sma20) and (close < sma20 * 0.992) and (rsi14 < 45.0):
        sell_signals.append("⚠️ [매도-추세이탈: 20일 이동평균선 하향 붕괴]")
        sell_reasons.append(
            f"종가(${close:.2f})가 20일 생명선(${sma20:.2f})을 하향 이탈하고 모멘텀 약화(RSI14 {rsi14:.1f}) — 스윙 포지션 정리/손절 권장"
        )

    # =========================================================================
    # ATR 기반 추천 손절가(Stop Loss) 및 전략별 맞춤 익절 가이드 생성
    # =========================================================================
    stop_loss = round(close - (atr * 2.2), 2)
    target_1 = round(close + (atr * STRATEGY_PARAMS["atr_target_1_multiplier"]), 2)
    target_2 = round(close + (atr * STRATEGY_PARAMS["atr_target_2_multiplier"]), 2)

    if any("과매도" in s for s in signals):
        exit_guide = (
            f"단기 평균회귀 전략 — 1차 목표: 5일 이동평균선(${sma5:.2f}) 회복 시 단기 익절 / "
            f"2차 목표: 20일선(${sma20:.2f}) 도달 시 전량 익절"
        )
    elif any("돌파" in s for s in signals):
        exit_guide = (
            f"추세 돌파 전략 — 1차 목표(${target_1:.2f})에서 50% 분할익절 후, 잔여 물량은 고점 대비 -2.5 ATR(${round(close - atr*2.5, 2):.2f}) "
            f"트레일링 스탑 또는 20일선(${sma20:.2f}) 이탈 전까지 홀딩"
        )
    else:
        exit_guide = f"5일선 ${sma5:.2f} | 20일선 ${sma20:.2f} | ATR 손절선 ${stop_loss:.2f}"

    # 현재 차트 위치 요약
    if close >= sma20 and close >= sma50:
        trend_str = "강세 구간 (20일·50일선 위)"
    elif close >= sma20 and close < sma50:
        trend_str = "단기 반등 중 (20일선 회복, 50일선 아래)"
    elif close < sma20 and close >= sma50:
        trend_str = "단기 눌림목 (20일선 아래, 50일선 위)"
    elif close >= sma200:
        trend_str = "중기 조정 중 (50일선 아래, 200일선 위)"
    else:
        trend_str = "약세 구간 (200일선 하회)"

    watch_status = f"{trend_str} | 20일선 이격 {dist_sma20_pct:+.1f}% | RSI(14) {rsi14:.1f} | RSI(2) {rsi2:.1f}"

    indicators_summary = {
        "SMA5": round(sma5, 2),
        "SMA20": round(sma20, 2),
        "SMA50": round(sma50, 2),
        "SMA200": round(sma200, 2),
        "dist_sma20_pct": round(dist_sma20_pct, 2),
        "dist_sma50_pct": round(dist_sma50_pct, 2),
        "RSI14": round(rsi14, 1),
        "RSI2": round(rsi2, 1),
        "Vol_Ratio": round(vol_ratio, 2),
        "ATR14": round(atr, 2),
        "BB_Lower": round(bb_lower, 2),
        "BB_Upper": round(bb_upper, 2),
        "Prev_High_20": round(prev_high_close, 2),
    }

    return SwingSignalResult(
        ticker=ticker,
        name=name,
        category=category,
        current_price=round(close, 2),
        change_pct=round(change_pct, 2),
        signals=signals,
        reasons=reasons,
        sell_signals=sell_signals,
        sell_reasons=sell_reasons,
        exit_guide=exit_guide,
        indicators=indicators_summary,
        stop_loss=stop_loss,
        target_1=target_1,
        target_2=target_2,
        watch_status=watch_status,
    )
