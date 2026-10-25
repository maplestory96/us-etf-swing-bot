"""
미국 ETF 스윙 시그널 봇 - 종목 성격별 스마트 라우팅(Smart Strategy Routing) 5년 최종 백테스팅 엔진
검증 대상:
  [OVERSOLD_ONLY 그룹] QQQ (나스닥100), SOXX (반도체1x), SOXL (반도체3x), TQQQ (나스닥3x)
    -> 가짜 돌파 원천 배제, 승률 80%+ '💎과매도 반등' 스나이핑 전용
  [MACRO_CYCLE 그룹] XLE (에너지), GLD (금 현물)
    -> '🔥거래량 폭발 박스권 돌파' 주력 + '💎극단적 과매도 반등(RSI2<=10)' 병행
"""
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from config import STRATEGY_PARAMS, ETF_UNIVERSE
from data_fetcher import fetch_ohlcv
from technical_analyzer import compute_indicators

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass
class TradeRecord:
    ticker: str
    strategy: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    holding_days: int
    exit_reason: str


def check_smart_routed_signals(curr: pd.Series, prev: pd.Series, routing_mode: str) -> list[str]:
    close = float(curr["Close"])
    open_p = float(curr["Open"])
    prev_close = float(prev["Close"])

    sma20 = float(curr["SMA20"])
    sma50 = float(curr["SMA50"])
    sma200 = float(curr["SMA200"]) if not pd.isna(curr["SMA200"]) else sma50
    sma20_slope = float(curr["SMA20_Slope"]) if not pd.isna(curr["SMA20_Slope"]) else 0.0
    sma200_slope = float(curr["SMA200_Slope"]) if not pd.isna(curr["SMA200_Slope"]) else 0.0

    rsi14 = float(curr["RSI14"])
    prev_rsi14 = float(prev["RSI14"])
    rsi2 = float(curr["RSI2"])
    vol_ratio = float(curr["Vol_Ratio"]) if not pd.isna(curr["Vol_Ratio"]) else 1.0
    bb_lower = float(curr["BB_Lower"])
    macd_hist = float(curr["MACD_Hist"])
    prev_macd_hist = float(prev["MACD_Hist"])
    prev_high_close = float(curr["Prev_Close_High_N"])

    if np.isnan(sma50) or np.isnan(prev_high_close):
        return []

    found = []

    # 1) OVERSOLD_ONLY 그룹 (QQQ, SOXX, SOXL, TQQQ 등): 과매도 반등 스나이핑 100% 집중
    if routing_mode == "OVERSOLD_ONLY":
        rsi14_thresh = STRATEGY_PARAMS["oversold_rsi14_threshold"]
        rsi2_thresh = STRATEGY_PARAMS["oversold_rsi2_threshold"]
        cond_oversold_std = (
            (rsi14 <= rsi14_thresh or prev_rsi14 <= rsi14_thresh)
            and (close <= bb_lower * 1.015)
            and (close > open_p or rsi14 > prev_rsi14)
        )
        cond_oversold_rsi2 = (rsi2 <= rsi2_thresh) and (close > sma200 * 0.96) and (sma200_slope >= -0.5)
        if cond_oversold_std or cond_oversold_rsi2:
            found.append("💎과매도반등(스나이핑)")

    # 2) MACRO_CYCLE 그룹 (XLE, GLD 등): 진성 거래량 돌파 주력 + 극단적 과매도(볼린저 하단 반등) 결합
    elif routing_mode == "MACRO_CYCLE":
        vol_thresh = STRATEGY_PARAMS["breakout_volume_ratio"]
        candle_range = max(float(curr["High"]) - float(curr["Low"]), 0.0001)
        close_strength = (close - float(curr["Low"])) / candle_range

        is_price_breakout = (close > prev_high_close) and (prev_close <= prev_high_close * 1.002)
        is_trend_aligned = (close > sma50) and (sma20_slope > 0) and (50.0 <= rsi14 < 70.0)
        is_strong_breakout_candle = (close > open_p) and (close_strength >= 0.6)
        is_volume_and_momentum = (vol_ratio >= vol_thresh) and (macd_hist > 0) and (macd_hist > prev_macd_hist)

        if is_price_breakout and is_trend_aligned and is_strong_breakout_candle and is_volume_and_momentum:
            found.append("🔥박스권돌파(거래량폭발)")

        # 매크로 자산이 장기 추세(200일선) 위에서 볼린저 하단 + 극단적 과매도(RSI14<=32) 도달 후 양봉 반등 시
        cond_macro_oversold = (
            (close > sma200)
            and (rsi14 <= 32.0 or prev_rsi14 <= 32.0)
            and (close <= bb_lower * 1.01)
            and (close > open_p)
        )
        if cond_macro_oversold:
            found.append("💎과매도반등(매크로지지)")

    return found


def simulate_smart_swing(
    ticker: str,
    d: pd.DataFrame,
    routing_mode: str,
    commission_pct: float = 0.1,
) -> tuple[list[TradeRecord], pd.Series]:
    trades: list[TradeRecord] = []

    in_position = False
    entry_idx = 0
    entry_date = ""
    entry_price = 0.0
    initial_stop = 0.0
    trailing_stop = 0.0
    highest_since_entry = 0.0
    active_strategy = ""

    equity_curve = pd.Series(1.0, index=d.index, dtype=float)
    current_equity = 1.0

    start_i = min(100, len(d) // 4)

    for i in range(start_i, len(d)):
        curr = d.iloc[i]
        prev = d.iloc[i - 1]
        date_str = d.index[i].strftime("%Y-%m-%d")

        if not in_position:
            equity_curve.iloc[i] = current_equity
            sigs = check_smart_routed_signals(curr, prev, routing_mode=routing_mode)
            if sigs:
                in_position = True
                entry_idx = i
                entry_date = date_str
                entry_price = float(curr["Close"])
                highest_since_entry = float(curr["High"])
                atr = float(curr["ATR14"]) if not pd.isna(curr["ATR14"]) else entry_price * 0.02
                active_strategy = sigs[0]

                initial_stop = entry_price - (atr * 2.2)
                trailing_stop = initial_stop
        else:
            hold_days = i - entry_idx
            high_p = float(curr["High"])
            low_p = float(curr["Low"])
            close_p = float(curr["Close"])
            sma5 = float(curr["SMA5"])
            sma20 = float(curr["SMA20"])
            rsi2 = float(curr["RSI2"])
            rsi14 = float(curr["RSI14"])
            atr = float(curr["ATR14"]) if not pd.isna(curr["ATR14"]) else close_p * 0.02

            if high_p > highest_since_entry:
                highest_since_entry = high_p
            if highest_since_entry >= entry_price + (atr * 1.5):
                new_trail = highest_since_entry - (atr * 2.5)
                if new_trail > trailing_stop:
                    trailing_stop = new_trail

            exit_triggered = False
            exit_price = close_p
            exit_reason = ""

            if "과매도반등" in active_strategy:
                if close_p > sma5 and rsi2 >= 65 and close_p > entry_price:
                    exit_triggered = True
                    exit_price = close_p
                    exit_reason = "반등익절(5일선회복)"
                elif low_p <= initial_stop:
                    exit_triggered = True
                    exit_price = min(float(curr["Open"]), initial_stop)
                    exit_reason = "손절(지지이탈)"
                elif hold_days >= 12:
                    exit_triggered = True
                    exit_price = close_p
                    exit_reason = "기간청산(12일)"
            else:
                if low_p <= trailing_stop:
                    exit_triggered = True
                    exit_price = min(float(curr["Open"]), trailing_stop) if float(curr["Open"]) < trailing_stop else trailing_stop
                    exit_reason = "트레일링익절" if exit_price > entry_price else "손절(추세이탈)"
                elif hold_days >= 10 and close_p < sma20 and rsi14 < 45:
                    exit_triggered = True
                    exit_price = close_p
                    exit_reason = "20일선마감이탈"
                elif hold_days >= 45:
                    exit_triggered = True
                    exit_price = close_p
                    exit_reason = "장기스윙만료(45일)"

            if exit_triggered:
                raw_ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
                net_ret_pct = raw_ret_pct - commission_pct
                current_equity *= 1.0 + (net_ret_pct / 100.0)
                equity_curve.iloc[i] = current_equity

                trades.append(
                    TradeRecord(
                        ticker=ticker,
                        strategy=active_strategy,
                        entry_date=entry_date,
                        exit_date=date_str,
                        entry_price=round(entry_price, 2),
                        exit_price=round(exit_price, 2),
                        return_pct=round(net_ret_pct, 2),
                        holding_days=hold_days,
                        exit_reason=exit_reason,
                    )
                )
                in_position = False
            else:
                unrealized_ret = (close_p - entry_price) / entry_price
                equity_curve.iloc[i] = current_equity * (1.0 + unrealized_ret)

    return trades, equity_curve.iloc[start_i:]


def calc_mdd(series: pd.Series) -> float:
    roll_max = series.cummax()
    drawdown = (series - roll_max) / roll_max * 100.0
    return float(drawdown.min())


def run_backtest():
    test_tickers = ["QQQ", "SOXX", "SOXL", "TQQQ", "XLE", "GLD"]

    print("=" * 95)
    print("🏆 [미국 ETF 6대장 - 종목 성격별 스마트 라우팅(Smart Routing) 5년 최종 백테스트 성적표]")
    print("  • 지수/반도체/3배레버리지 (QQQ, SOXX, SOXL, TQQQ) 👉 [💎과매도 반등 스나이핑 전용]")
    print("  • 에너지/원자재/매크로 (XLE, GLD)                 👉 [🔥거래량 폭발 돌파 + 💎장기지지 과매도]")
    print("=" * 95)
    print(
        f"  {'종목(티커)':<18} | {'적용 라우팅 모드':<15} | {'매매수':>5} | {'승률':>7} | "
        f"{'스윙수익률':>9} | {'스윙MDD':>8} | {'존버MDD':>8} | {'보유일':>5}"
    )
    print("-" * 95)

    all_trades: list[TradeRecord] = []

    for ticker in test_tickers:
        meta = ETF_UNIVERSE[ticker]
        routing_mode = meta["mode"]
        df = fetch_ohlcv(ticker, period="5y")
        if df.empty:
            continue
        d = compute_indicators(df)

        start_i = min(100, len(d) // 4)
        bh_series = d["Close"].iloc[start_i:]
        bh_mdd = calc_mdd(bh_series)

        trades, eq_curve = simulate_smart_swing(ticker, d, routing_mode=routing_mode)
        all_trades.extend(trades)

        cnt = len(trades)
        wins = [t for t in trades if t.return_pct > 0]
        win_rate = (len(wins) / cnt * 100.0) if cnt > 0 else 0.0
        cum_ret = (eq_curve.iloc[-1] - 1.0) * 100.0
        mdd = calc_mdd(eq_curve)
        avg_hold = np.mean([t.holding_days for t in trades]) if cnt > 0 else 0.0

        print(
            f"  {ticker:<6} ({meta['category'][:8]:<8}) | {routing_mode:<15} | {cnt:>4}회 | {win_rate:>6.1f}% | "
            f"{cum_ret:>+8.1f}% | {mdd:>7.1f}% | {bh_mdd:>7.1f}% | {avg_hold:>4.1f}일"
        )

    print("-" * 95)
    trades_df = pd.DataFrame([t.__dict__ for t in all_trades])
    if not trades_df.empty:
        total_cnt = len(trades_df)
        total_w = len(trades_df[trades_df["return_pct"] > 0])
        print(
            f"  🌟 6개 대표 ETF 전체 통합: 총 {total_cnt}회 매매 | 평균 승률: {total_w / total_cnt * 100:.1f}% | "
            f"1회당 평균 기대수익: {trades_df['return_pct'].mean():+.2f}% | 평균 보유기간: {trades_df['holding_days'].mean():.1f}거래일"
        )
    print("=" * 95)


if __name__ == "__main__":
    run_backtest()
