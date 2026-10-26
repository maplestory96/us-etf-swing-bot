r"""
미국 ETF 스윙 시그널 봇 - 메인 실행 파일 (main.py)
실행 방법:
  1회 즉시 스캔: .\.venv\Scripts\python.exe main.py
  특정 종목만 스캔: .\.venv\Scripts\python.exe main.py --tickers SPY QQQ SOXX SOXL TQQQ XLE GLD
  주기적 자동 반복 실행 (예: 60분마다 무한 감시): .\.venv\Scripts\python.exe main.py --loop 60
"""
import argparse
import sys
import time
from datetime import datetime
from config import ETF_UNIVERSE, KOREAN_ETF_UNIVERSE
from data_fetcher import fetch_ohlcv, fetch_etf_news
from technical_analyzer import analyze_etf_chart, SwingSignalResult
from korean_analyzer import analyze_korean_etf, KoreanSwingSignal
from ai_news_analyzer import evaluate_signal_with_ai, AINewsVerdict
from position_manager import load_positions, save_positions, update_positions_after_scan
from notifier import send_telegram_message

# Windows 터미널 한글/이모지 출력 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def format_report(
    buy_items: list[tuple[SwingSignalResult, AINewsVerdict]],
    sell_items: list[tuple[SwingSignalResult, AINewsVerdict]],
    all_results: list[SwingSignalResult],
    active_positions: dict,
) -> str:
    """매수 시그널과 매도(익절/손절/AI조기탈출) 시그널을 모두 포함한 미국 ETF 텔레그램 브리핑 메시지를 생성합니다."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"📊 [미국 ETF 스윙 시그널 AI 리포트 (Smart Routing)]",
        f"⏰ 스캔 시각: {now_str} (KST)",
        f"🔎 스캔 대상: 총 {len(all_results)}개 핵심 ETF (5년 검증 승률 80.3%)",
        f"🎯 포착 시그널: 매수 {len(buy_items)}건 / 매도(청산) {len(sell_items)}건",
        "=" * 42,
    ]

    # 1. 매수(진입) 시그널 섹션
    lines.append(f"\n🟢 [1. 신규 매수 타점 & AI 뉴스 심사: {len(buy_items)}건]")
    if buy_items:
        for idx, (sig, ai_res) in enumerate(buy_items, 1):
            veto_mark = "⛔ [AI 진입보류 권고]" if ai_res.action_type == "BUY_VETO" else f"✅ [권장 비중: {ai_res.recommended_weight_pct}%]"
            lines.append(f"\n[{idx}] 📌 {sig.ticker} ({sig.name}) — {veto_mark}")
            lines.append(f"• 현재가: ${sig.current_price:.2f} ({sig.change_pct:+.2f}%)")
            for s_tag in sig.signals:
                lines.append(f"• 차트 시그널: {s_tag}")
            for r_text in sig.reasons:
                lines.append(f"• 차트 근거: {r_text}")

            stop_pct = ((sig.stop_loss - sig.current_price) / sig.current_price) * 100
            t1_pct = ((sig.target_1 - sig.current_price) / sig.current_price) * 100
            sma5_val = sig.indicators.get("SMA5", sig.current_price)

            lines.append(
                f"• 가격 & 청산 가이드:"
                f"\n   ├ 권장 손절가: ${sig.stop_loss:.2f} ({stop_pct:+.1f}%)"
                f"\n   ├ 단기 5일선(과매도 반등 익절선): ${sma5_val:.2f}"
                f"\n   ├ 추세 1차 목표가(+2.5 ATR): ${sig.target_1:.2f} ({t1_pct:+.1f}%)"
                f"\n   └ 💡 맞춤 전략: {sig.exit_guide}"
            )
            lines.append(
                f"• 🤖 AI 뉴스 매수 보조 판단: {ai_res.verdict} (뉴스감성: {ai_res.sentiment_score:+d}/5)"
            )
            lines.append(f"   ├ 뉴스 요약: {ai_res.news_summary_kr}")
            lines.append(f"   └ AI 행동 지침: {ai_res.swing_advice_kr}")
            lines.append("-" * 42)
    else:
        lines.append("  • 현재 신규 매수 진입 조건에 부합하는 미국 ETF가 없습니다.")

    # 2. 매도 시그널 섹션
    lines.append(f"\n🔴 [2. 매도(익절/손절) & AI 뉴스 청산 심사: {len(sell_items)}건]")
    if sell_items:
        for idx, (sig, ai_res) in enumerate(sell_items, 1):
            lines.append(f"\n[{idx}] 🚨 {sig.ticker} ({sig.name}) — [권장 매도비중: {ai_res.recommended_weight_pct}%]")
            lines.append(f"• 현재가: ${sig.current_price:.2f} ({sig.change_pct:+.2f}%)")
            for s_tag in sig.sell_signals:
                lines.append(f"• 매도 시그널: {s_tag}")
            for r_text in sig.sell_reasons:
                lines.append(f"• 매도 근거: {r_text}")
            lines.append(
                f"• 🤖 AI 뉴스 매도 보조 판단: {ai_res.verdict} (뉴스감성: {ai_res.sentiment_score:+d}/5)"
            )
            lines.append(f"   ├ 뉴스 요약: {ai_res.news_summary_kr}")
            lines.append(f"   └ AI 행동 지침: {ai_res.swing_advice_kr}")
            lines.append("-" * 42)
    else:
        lines.append("  • 현재 매도(익절/손절/추세이탈/악재경보) 시그널이 발생한 종목이 없습니다.")

    # 3. 포지션 현황
    if active_positions:
        lines.append(f"\n💼 [현재 자동 추적 중인 미국 스윙 포지션: {len(active_positions)}건]")
        for tk, pinfo in active_positions.items():
            w_pct = pinfo.get("weight_pct", 100)
            lines.append(
                f"  • {tk} (비중 {w_pct}% | {pinfo.get('strategy', '')}): 진입가 ${pinfo.get('entry_price')} "
                f"({pinfo.get('entry_date')}) | 손절선 ${pinfo.get('stop_loss')} / 목표 ${pinfo.get('target_1')}"
            )

    # 4. 차트 요약
    lines.append("\n📋 [주요 미국 ETF 차트 상태 요약]")
    for r in all_results[:8]:
        mark = "🟢" if r.signals else ("🔴" if r.sell_signals else "•")
        lines.append(
            f"{mark} {r.ticker} (${r.current_price:.2f}, {r.change_pct:+.1f}%): {r.watch_status}"
        )

    return "\n".join(lines)


def format_kr_report(
    buy_items: list[tuple[KoreanSwingSignal, AINewsVerdict | None]],
    sell_items: list[tuple[KoreanSwingSignal, AINewsVerdict | None]],
    all_results: list[KoreanSwingSignal],
    active_positions: dict,
) -> str:
    """국내(한국장) ETF 11종 맞춤형 챔피언 전략 리포트를 생성합니다."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"🇰🇷 [국내 ETF 챔피언 스윙 시그널 AI 리포트]",
        f"⏰ 스캔 시각: {now_str} (KST)",
        f"🔎 스캔 대상: 국장 대표 ETF 11종 (5년 백테스트 최적 챔피언 전략)",
        f"🎯 포착 시그널: 매수 {len(buy_items)}건 / 매도(청산) {len(sell_items)}건",
        "=" * 44,
    ]

    # 1. 매수 시그널 섹션
    lines.append(f"\n🟢 [1. 국장 챔피언 매수 타점: {len(buy_items)}건]")
    if buy_items:
        for idx, (sig, ai_res) in enumerate(buy_items, 1):
            veto_mark = ""
            if ai_res:
                veto_mark = " ⛔ [AI 진입보류]" if ai_res.action_type == "BUY_VETO" else f" ✅ [권장비중: {ai_res.recommended_weight_pct}%]"
            
            lines.append(f"\n[{idx}] 📌 {sig.ticker} ({sig.name}) — {sig.strategy_name}{veto_mark}")
            lines.append(f"• 현재가: ₩{sig.current_price:,.0f} ({sig.change_pct:+.2f}%)")
            for s_tag in sig.signals:
                lines.append(f"• 타점 시그널: {s_tag}")
            for r_text in sig.reasons:
                lines.append(f"• 차트 근거: {r_text}")

            stop_pct = ((sig.stop_loss - sig.current_price) / sig.current_price) * 100
            t1_pct = ((sig.target_1 - sig.current_price) / sig.current_price) * 100

            lines.append(
                f"• 가격 & 청산 가이드:"
                f"\n   ├ 권장 손절가: ₩{sig.stop_loss:,.0f} ({stop_pct:+.1f}%)"
                f"\n   ├ 1차 목표가: ₩{sig.target_1:,.0f} ({t1_pct:+.1f}%)"
                f"\n   ├ 2차 목표가: ₩{sig.target_2:,.0f}"
                f"\n   └ 💡 챔피언 청산 원칙: {sig.exit_guide}"
            )
            if ai_res:
                lines.append(f"• 🤖 AI 뉴스 보조 판단: {ai_res.verdict}")
                lines.append(f"   ├ 뉴스 요약: {ai_res.news_summary_kr}")
                lines.append(f"   └ 행동 지침: {ai_res.swing_advice_kr}")
            lines.append("-" * 44)
    else:
        lines.append("  • 현재 엄격한 국장 챔피언 매수 기준에 부합하는 ETF가 없습니다.")

    # 2. 매도 시그널 섹션
    lines.append(f"\n🔴 [2. 국장 보유/매도(익절/손절) 심사: {len(sell_items)}건]")
    if sell_items:
        for idx, (sig, ai_res) in enumerate(sell_items, 1):
            lines.append(f"\n[{idx}] 🚨 {sig.ticker} ({sig.name}) — [청산/익절 경보]")
            lines.append(f"• 현재가: ₩{sig.current_price:,.0f} ({sig.change_pct:+.2f}%)")
            for s_tag in sig.sell_signals:
                lines.append(f"• 매도 시그널: {s_tag}")
            for r_text in sig.sell_reasons:
                lines.append(f"• 매도 근거: {r_text}")
            lines.append("-" * 44)
    else:
        lines.append("  • 현재 매도(익절/손절/이탈) 시그널이 발생한 종목이 없습니다.")

    # 3. 국장 포지션 현황
    if active_positions:
        lines.append(f"\n💼 [현재 자동 추적 중인 국장 스윙 포지션: {len(active_positions)}건]")
        for tk, pinfo in active_positions.items():
            lines.append(
                f"  • {tk} ({pinfo.get('name', '')}): 진입가 ₩{pinfo.get('entry_price', 0):,.0f} "
                f"({pinfo.get('entry_date')}) | 손절선 ₩{pinfo.get('stop_loss', 0):,.0f} / 목표 ₩{pinfo.get('target_1', 0):,.0f}"
            )

    # 4. 차트 위치 요약
    lines.append("\n📋 [국내 ETF 11종 현재 차트 상태 요약]")
    for r in all_results:
        mark = "🟢" if r.signals else ("🔴" if r.sell_signals else "•")
        lines.append(
            f"{mark} {r.name} (₩{r.current_price:,.0f}, {r.change_pct:+.1f}%): {r.watch_status}"
        )

    return "\n".join(lines)


def run_scanner(target_tickers: list[str] | None = None):
    """미국 ETF 스캐너 실행"""
    universe = ETF_UNIVERSE
    if target_tickers:
        universe = {
            k: v for k, v in ETF_UNIVERSE.items() if k.upper() in [t.upper() for t in target_tickers]
        }
        for t in target_tickers:
            t_up = t.upper()
            if t_up not in universe:
                universe[t_up] = {"name": f"{t_up} ETF", "category": "사용자지정", "mode": "OVERSOLD_ONLY"}

    positions = load_positions(market="us")
    print(f"🚀 총 {len(universe)}개 미국 ETF 차트 데이터 수집 및 AI 매수·매도 보조 분석을 시작합니다...\n")

    all_results: list[SwingSignalResult] = []
    buy_items: list[tuple[SwingSignalResult, AINewsVerdict]] = []
    sell_items: list[tuple[SwingSignalResult, AINewsVerdict]] = []

    for ticker, meta in universe.items():
        print(f"  -> [{ticker}] {meta['name']} 분석 중...", end=" ", flush=True)
        df = fetch_ohlcv(ticker, period="1y")
        if df.empty or len(df) < 60:
            print("데이터 부족 (건너뜀)")
            continue

        active_pos = positions.get(ticker)
        strat_mode = meta.get("mode", "OVERSOLD_ONLY")
        sig = analyze_etf_chart(
            ticker,
            meta["name"],
            meta["category"],
            df,
            active_pos=active_pos,
            strategy_mode=strat_mode,
        )
        all_results.append(sig)

        buy_ai: AINewsVerdict | None = None
        sell_ai: AINewsVerdict | None = None
        news_items: list[dict] = []

        if sig.signals:
            news_items = fetch_etf_news(ticker, meta["name"], max_items=5)
            buy_ai = evaluate_signal_with_ai(sig, news_items, context_mode="BUY")
            print(f"🟢 매수 타점! -> AI 판정: {buy_ai.verdict}")
            buy_items.append((sig, buy_ai))

        if sig.sell_signals or active_pos:
            if not news_items:
                news_items = fetch_etf_news(ticker, meta["name"], max_items=5)
            sell_ai = evaluate_signal_with_ai(sig, news_items, context_mode="SELL")

            if active_pos and not sig.sell_signals and sell_ai.action_type == "EARLY_EXIT":
                sig.sell_signals.append("🚨 [AI 뉴스 긴급 매도: 보유종목 심각한 악재 감지]")
                sig.sell_reasons.append(
                    f"차트 손절선(${active_pos.get('stop_loss')}) 도달 전이나 뉴스 악재로 선제 탈출 권고"
                )

            if sig.sell_signals:
                print(f"🔴 매도 타점! -> AI 판정: {sell_ai.verdict}")
                sell_items.append((sig, sell_ai))

        if not sig.signals and not sig.sell_signals:
            print(f"대기 ({sig.watch_status})")

        update_positions_after_scan(positions, sig, buy_ai=buy_ai, sell_ai=sell_ai)

    save_positions(positions, market="us")

    print("\n" + "=" * 58)
    report_msg = format_report(buy_items, sell_items, all_results, positions)
    print(report_msg)
    print("=" * 58)

    send_telegram_message(report_msg)


def run_kr_scanner(target_tickers: list[str] | None = None):
    """국내(한국장) ETF 11종 챔피언 전략 스캐너 실행"""
    universe = KOREAN_ETF_UNIVERSE
    if target_tickers:
        universe = {
            k: v for k, v in KOREAN_ETF_UNIVERSE.items() if k in target_tickers or k.split(".")[0] in target_tickers
        }

    positions = load_positions(market="kr")
    print(f"🚀 총 {len(universe)}개 국내 ETF 챔피언 전략 분석 및 AI 시황 진단을 시작합니다...\n")

    all_results: list[KoreanSwingSignal] = []
    buy_items: list[tuple[KoreanSwingSignal, AINewsVerdict | None]] = []
    sell_items: list[tuple[KoreanSwingSignal, AINewsVerdict | None]] = []

    for ticker, meta in universe.items():
        print(f"  -> [{ticker}] {meta['name']} ({meta['category']}) 분석 중...", end=" ", flush=True)
        df = fetch_ohlcv(ticker, period="1y")
        if df.empty or len(df) < 60:
            print("데이터 부족 (건너뜀)")
            continue

        active_pos = positions.get(ticker)
        sig = analyze_korean_etf(
            ticker=ticker,
            name=meta["name"],
            category=meta["category"],
            strategy_code=meta["strategy"],
            df=df,
            active_pos=active_pos,
        )
        all_results.append(sig)

        buy_ai: AINewsVerdict | None = None
        sell_ai: AINewsVerdict | None = None

        if sig.signals:
            news_items = fetch_etf_news(ticker, meta["name"], max_items=4)
            # 한국 시장 뉴스가 있으면 AI 진입 심사
            if news_items:
                buy_ai = evaluate_signal_with_ai(sig, news_items, context_mode="BUY")
                print(f"🟢 매수 타점! -> AI 판정: {buy_ai.verdict}")
            else:
                print(f"🟢 매수 타점 포착! ({sig.strategy_name})")
            buy_items.append((sig, buy_ai))

        if sig.sell_signals or active_pos:
            news_items = fetch_etf_news(ticker, meta["name"], max_items=4)
            if news_items:
                sell_ai = evaluate_signal_with_ai(sig, news_items, context_mode="SELL")
            if sig.sell_signals:
                print(f"🔴 매도 타점! ({sig.sell_signals[0]})")
                sell_items.append((sig, sell_ai))

        if not sig.signals and not sig.sell_signals:
            print(f"대기 ({sig.watch_status})")

        update_positions_after_scan(positions, sig, buy_ai=buy_ai, sell_ai=sell_ai)

    save_positions(positions, market="kr")

    print("\n" + "=" * 58)
    report_msg = format_kr_report(buy_items, sell_items, all_results, positions)
    print(report_msg)
    print("=" * 58)

    send_telegram_message(report_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ETF 스윙 시그널 AI 봇 (미국장 + 국장 챔피언 전략 통합)")
    parser.add_argument(
        "--market",
        choices=["all", "us", "kr"],
        default="all",
        help="스캔할 시장 선택: 'us'(미국장), 'kr'(국내장), 'all'(둘 다 실행)",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="특정 ETF 티커만 지정하여 스캔 (예: --tickers 069500.KS 122630.KS 또는 QQQ SPY)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="지정한 분(minute) 간격으로 계속 스캔 반복 실행 (예: --loop 60 은 60분마다 자동 실행)",
    )
    args = parser.parse_args()

    def execute_scan():
        if args.market in ["kr", "all"]:
            run_kr_scanner(args.tickers)
        if args.market in ["us", "all"]:
            run_scanner(args.tickers)

    if args.loop > 0:
        print(f"🔄 [자동 반복 모드] {args.loop}분 간격으로 ETF 스윙 시그널 감시를 시작합니다. (시장: {args.market}, 종료: Ctrl+C)")
        while True:
            try:
                execute_scan()
                print(f"\n⏳ 다음 스캔까지 {args.loop}분 대기합니다...")
                time.sleep(args.loop * 60)
            except KeyboardInterrupt:
                print("\n🛑 사용자 요청으로 자동 감시를 종료합니다.")
                break
    else:
        execute_scan()

