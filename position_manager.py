"""
미국 ETF 스윙 시그널 봇 - 보유 포지션(가상/실전) 자동 추적 및 관리 모듈 (position_manager.py)
AI 뉴스 판단(매수 거부권 BUY_VETO, 홀딩 연장 HOLD_EXTEND, 조기 매도 EARLY_EXIT)까지 반영하여 포지션을 관리합니다.
"""
import json
import os
from datetime import datetime
from technical_analyzer import SwingSignalResult
from ai_news_analyzer import AINewsVerdict

POSITIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions.json")


def load_positions() -> dict:
    if not os.path.exists(POSITIONS_FILE):
        return {}
    try:
        with open(POSITIONS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_positions(positions: dict) -> None:
    try:
        with open(POSITIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(positions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[경고] 포지션 파일 저장 실패: {e}")


def update_positions_after_scan(
    positions: dict,
    sig: SwingSignalResult,
    buy_ai: AINewsVerdict | None = None,
    sell_ai: AINewsVerdict | None = None,
) -> None:
    """
    1) 보유 중인 종목:
       - 매도 시그널 발생 시 AI가 'HOLD_EXTEND'(호재로 50%만 익절 후 홀딩 연장) 판정을 내리면 포지션의 50%를 유지하고 손절선을 본절(진입가)로 상향!
       - 일반 매도('SELL_NOW') 또는 AI 조기 탈출('EARLY_EXIT') 판정이면 포지션 전량 청산 완료 처리
    2) 미보유 종목:
       - 차트 매수 시그널이 발생했더라도 AI가 'BUY_VETO'(대형 악재로 진입 보류) 판정을 내리면 진입하지 않고 패스!
       - 'BUY_STRONG' 또는 'BUY_NORMAL' 승인 시에만 포지션 등록
    """
    tk = sig.ticker
    today_str = datetime.now().strftime("%Y-%m-%d")

    if tk in positions:
        if sig.sell_signals:
            if sell_ai and sell_ai.action_type == "HOLD_EXTEND" and not positions[tk].get("extended_once"):
                # 호재 동반으로 절반만 익절하고 나머지는 본절 스탑으로 홀딩 연장!
                positions[tk]["extended_once"] = True
                positions[tk]["stop_loss"] = positions[tk]["entry_price"]  # 손절선을 진입가(본절)로 상향
                positions[tk]["weight_pct"] = 50
            else:
                # 전량 청산 완료
                del positions[tk]
        else:
            prev_high = float(positions[tk].get("highest_price", sig.current_price))
            if sig.current_price > prev_high:
                positions[tk]["highest_price"] = sig.current_price
    else:
        if sig.signals:
            # AI가 악재로 매수 거부권(BUY_VETO)을 발동한 경우 포지션에 편입하지 않음!
            if buy_ai and buy_ai.action_type == "BUY_VETO":
                return

            weight = buy_ai.recommended_weight_pct if buy_ai else 70
            positions[tk] = {
                "name": sig.name,
                "entry_date": today_str,
                "entry_price": sig.current_price,
                "highest_price": sig.current_price,
                "stop_loss": sig.stop_loss,
                "target_1": sig.target_1,
                "target_2": sig.target_2,
                "strategy": sig.signals[0],
                "weight_pct": weight,
                "extended_once": False,
            }
