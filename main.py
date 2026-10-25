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
from config import ETF_UNIVERSE
from data_fetcher import fetch_ohlcv, fetch_etf_news
from technical_analyzer import analyze_etf_chart, SwingSignalResult
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
    """매수 시그널과 매도(익절/손절/AI조기탈출) 시그널을 모두 포함한 텔레그램 브리핑 메시지를 생성합니다."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"📊 [미국 ETF 스윙 시그널 AI 리포트 (Smart Routing)]",
        f"⏰ 스캔 시각: {now_str} (KST)",
        f"🔎 스캔 대상: 총 {len(all_results)}개 핵심 ETF (5년 검증 승률 80.3%)",
        f"🎯 포착 시그널: 매수 {len(buy_items)}건 / 매도(청산) {len(sell_items)}건",
        "=" * 42,
    ]

    # =========================================================================
    # 1. 매수(진입) 시그널 섹션 (차트 타점 + AI 뉴스 진입 심사/비중 조절)
    # =========================================================================
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
        lines.append("  • 현재 신규 매수 진입 조건에 부합하는 ETF가 없습니다.")

    # =========================================================================
    # 2. 매도(익절/트레일링스탑/손절/AI조기탈출) 시그널 섹션
    # =========================================================================
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

    # =========================================================================
    # 3. 현재 추적 중인 스윙 포지션 현황 (positions.json)
    # =========================================================================
    if active_positions:
        lines.append(f"\n💼 [현재 자동 추적 중인 스윙 포지션: {len(active_positions)}건]")
        for tk, pinfo in active_positions.items():
            w_pct = pinfo.get("weight_pct", 100)
            lines.append(
                f"  • {tk} (비중 {w_pct}% | {pinfo.get('strategy', '')}): 진입가 ${pinfo.get('entry_price')} "
                f"({pinfo.get('entry_date')}) | 손절선 ${pinfo.get('stop_loss')} / 목표 ${pinfo.get('target_1')}"
            )

    # =========================================================================
    # 4. 주요 ETF 차트 위치 요약
    # =========================================================================
    lines.append("\n📋 [주요 ETF 현재 차트 위치 요약]")
    for r in all_results[:8]:
        mark = "🟢" if r.signals else ("🔴" if r.sell_signals else "•")
        lines.append(
            f"{mark} {r.ticker} (${r.current_price:.2f}, {r.change_pct:+.1f}%): {r.watch_status}"
        )

    return "\n".join(lines)


def run_scanner(target_tickers: list[str] | None = None):
    universe = ETF_UNIVERSE
    if target_tickers:
        universe = {
            k: v for k, v in ETF_UNIVERSE.items() if k.upper() in [t.upper() for t in target_tickers]
        }
        for t in target_tickers:
            t_up = t.upper()
            if t_up not in universe:
                universe[t_up] = {"name": f"{t_up} ETF", "category": "사용자지정", "mode": "OVERSOLD_ONLY"}

    positions = load_positions()
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

        # 1) 차트 매수 시그널 발생 시 -> AI 뉴스 매수 심사 (비중 조절 / 매수 거부권)
        if sig.signals:
            news_items = fetch_etf_news(ticker, meta["name"], max_items=5)
            buy_ai = evaluate_signal_with_ai(sig, news_items, context_mode="BUY")
            print(f"🟢 매수 타점! -> AI 판정: {buy_ai.verdict}")
            buy_items.append((sig, buy_ai))

        # 2) 차트 매도 시그널 발생 시 OR 현재 보유 중인 포지션일 때 -> AI 뉴스 매도/보유 심사
        if sig.sell_signals or active_pos:
            if not news_items:
                news_items = fetch_etf_news(ticker, meta["name"], max_items=5)
            sell_ai = evaluate_signal_with_ai(sig, news_items, context_mode="SELL")

            # 보유 중 차트 매도 시그널이 없더라도 AI가 치명적 악재(EARLY_EXIT)를 감지하면 긴급 매도 시그널 발동!
            if active_pos and not sig.sell_signals and sell_ai.action_type == "EARLY_EXIT":
                sig.sell_signals.append("🚨 [AI 뉴스 긴급 매도: 보유종목 심각한 악재 감지]")
                sig.sell_reasons.append(
                    f"차트 손절선(${active_pos.get('stop_loss')}) 도달 전이나 뉴스 악재(감성 {sell_ai.sentiment_score}점)로 선제 탈출 권고"
                )

            if sig.sell_signals:
                print(f"🔴 매도 타점! -> AI 판정: {sell_ai.verdict}")
                sell_items.append((sig, sell_ai))

        if not sig.signals and not sig.sell_signals:
            print(f"대기 ({sig.watch_status})")

        # 스캔 및 AI 판정 결과에 따라 보유 포지션 자동 갱신
        update_positions_after_scan(positions, sig, buy_ai=buy_ai, sell_ai=sell_ai)

    save_positions(positions)

    print("\n" + "=" * 58)
    report_msg = format_report(buy_items, sell_items, all_results, positions)
    print(report_msg)
    print("=" * 58)

    send_telegram_message(report_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="미국 ETF 스윙 시그널 AI 봇 (매수/매도 AI 보조 통합)")
    parser.add_argument(
        "--tickers",
        nargs="*",
        help="특정 ETF 티커만 지정하여 스캔 (예: --tickers SPY QQQ SOXX)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="지정한 분(minute) 간격으로 계속 스캔 반복 실행 (예: --loop 60 은 60분마다 자동 실행)",
    )
    args = parser.parse_args()

    if args.loop > 0:
        print(f"🔄 [자동 반복 모드] {args.loop}분 간격으로 미국 ETF 스윙 시그널 감시를 시작합니다. (종료: Ctrl+C)")
        while True:
            try:
                run_scanner(args.tickers)
                print(f"\n⏳ 다음 스캔까지 {args.loop}분 대기합니다...")
                time.sleep(args.loop * 60)
            except KeyboardInterrupt:
                print("\n🛑 사용자 요청으로 자동 감시를 종료합니다.")
                break
    else:
        run_scanner(args.tickers)
