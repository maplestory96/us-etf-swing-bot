"""
미국 ETF 스윙 시그널 봇 - 멀티 AI 뉴스 감성 및 매수·매도 의사결정 보조 엔진 (ai_news_analyzer.py)
역할:
  1) [매수 보조]: 차트 매수 타점 포착 시 뉴스를 분석해
     - 호재 동반(+2~+5) -> '🟢 강력 매수 승인 (권장비중 100%)'
     - 중립/단순조정(-2~+1) -> '🟡 일반 매수 승인 (권장비중 50~70%)'
     - 대형 악재(-5~-3) -> '⛔ 진입 보류 / AI 매수 거부권 발동 (관망)'
  2) [매도 보조]: 차트 매도 타점 또는 보유 중인 포지션의 뉴스를 분석해
     - 익절 타점 + 강력 호재(+3~+5) -> '🔥 절반(50%)만 분할익절 후 홀딩 연장 권고'
     - 일반 익절/손절 타점 -> '💰 계획대로 전량 익절/청산 실행'
     - 보유 중 차트 손절 전이나 치명적 악재(-4 이하) -> '🚨 AI 긴급 조기 매도(선제 탈출) 경보'
"""
import json
import re
import requests
from typing import Any
from pydantic import BaseModel, Field
from config import AI_PROVIDER, LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, GEMINI_API_KEY
from technical_analyzer import SwingSignalResult


class AINewsVerdict(BaseModel):
    sentiment_score: int = Field(
        description="뉴스 및 매크로 감성 점수 (-5: 치명적 구조적 악재 ~ 0: 중립/단순기술적조정 ~ +5: 강력한 호재/서프라이즈)"
    )
    action_type: str = Field(
        description=(
            "최종 AI 보조 의사결정 코드: "
            "매수 시 ['BUY_STRONG'(강력매수 100%), 'BUY_NORMAL'(일반/분할매수 70%), 'BUY_CAUTION'(신중/비중축소 30~50%), 'BUY_VETO'(악재로 매수거부/보류 0%)] / "
            "매도·보유 시 ['SELL_NOW'(전량청산/익절), 'HOLD_EXTEND'(호재동반 분할익절 후 홀딩연장), 'EARLY_EXIT'(악재로 조기탈출), 'HOLD_KEEP'(정상보유유지)]"
        )
    )
    verdict: str = Field(
        description="텔레그램에 표시할 한 줄 행동 지침 (예: '🟢 매수 승인 (호재 동반 - 비중 100%)', '⛔ 매수 거부 (악재 경보 - 관망 권장)', '🔥 절반 익절 후 홀딩 연장 (강력 호재)')"
    )
    recommended_weight_pct: int = Field(
        description="매수 시 권장 투입 비중 (0, 30, 50, 70, 100 중 하나) 또는 매도 시 권장 매도 비중 (0, 50, 100 중 하나)"
    )
    bull_argument: str = Field(
        default="",
        description="상승론자(Bull)의 핵심 근거 1줄 (과매도 반발 매수세, 기술적 지지, 긍정 재료)"
    )
    bear_argument: str = Field(
        default="",
        description="하락론자(Bear)의 리스크 경고 1줄 (떨어지는 칼날 위험, 매크로/실적 악재, 가짜 반등 가능성)"
    )
    risk_assessment: str = Field(
        default="",
        description="리스크 심의 위원회의 최종 판정 이유 1줄 (거부권 Veto 발동 여부 및 비중 조절 사유)"
    )
    news_summary_kr: str = Field(
        description="최근 영문 뉴스 핵심 내용을 한국어로 1~2줄 요약"
    )
    swing_advice_kr: str = Field(
        description="차트 타점과 뉴스를 결합한 구체적인 매수/매도 보조 조언 (한국어 1~2줄)"
    )


# 룰 기반(키 미설정 시에도 작동하는) 영문 뉴스 핵심 키워드 감성 사전
POSITIVE_KEYWORDS = [
    "surge", "rally", "beat", "upgrade", "record high", "bull", "optimism",
    "rate cut", "dovish", "breakthrough", "strong demand", "inflow", "soar", "jump",
]
NEGATIVE_KEYWORDS = [
    "crash", "plunge", "tariff", "ban", "sanction", "recession", "inflation shock",
    "hawkish", "rate hike", "downgrade", "miss", "investigation", "lawsuit", "war", "default",
]


def _rule_based_news_score(news_items: list[dict]) -> int:
    """API 키가 없을 때도 뉴스 헤드라인 키워드로 기본적인 호재/악재를 판별하는 보조 함수"""
    if not news_items:
        return 0
    text = " ".join([(n.get("title", "") + " " + n.get("summary", "")).lower() for n in news_items])
    pos_hits = sum(1 for kw in POSITIVE_KEYWORDS if kw in text)
    neg_hits = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text)
    raw = pos_hits - (neg_hits * 2)  # 악재 키워드에 보수적 가중치 부여
    return max(-5, min(5, raw))


def _build_prompt(signal: SwingSignalResult, news_items: list[dict], context_mode: str) -> str:
    news_text = (
        "\n".join([f"- [{n['publisher']}] {n['title']} ({n['summary']})" for n in news_items])
        if news_items
        else "최근 특별한 뉴스 헤드라인 없음."
    )
    active_tags = signal.signals if context_mode == "BUY" else (signal.sell_signals or ["보유 포지션 뉴스 점검"])
    active_reasons = signal.reasons if context_mode == "BUY" else (signal.sell_reasons or [signal.watch_status])

    return f"""당신은 기관 헤지펀드의 [투자 심의 위원회(Investment Committee)]이자 수석 리스크 관리자(Chief Risk Officer)입니다.
TauricResearch TradingAgents 프레임워크처럼, 단일 판단이 아닌 세 가지 관점(Bull 토론, Bear 토론, 리스크 위원회 심의)을 거쳐 매매 결정을 철저히 검증합니다.

[1단계 퀀트 시그널]: 승률 80.3% 검증된 스마트 라우팅 기술적 타점이 포착되었습니다.
[2단계 리스크 심의]: 당신은 이 타점이 '진짜 반등'인지, 아니면 치명적 악재(블랙스완, 실적쇼크, 금리/전쟁)로 인한 '떨어지는 칼날'인지 뉴스/매크로를 토대로 교차 검증해야 합니다.

[분석 모드]: {context_mode} ({'신규 매수 진입 심사' if context_mode == 'BUY' else '매도 타점 및 보유 포지션 심사'})
[포착된 ETF 정보]
- 종목: {signal.ticker} ({signal.name} / {signal.category})
- 현재가: ${signal.current_price} (전일 대비 {signal.change_pct:+.2f}%)
- 포착된 차트 시그널: {', '.join(active_tags)}
- 차트 근거: {' / '.join(active_reasons)}
- 주요 지표: 5일선=${signal.indicators.get('SMA5')}, 20일선=${signal.indicators.get('SMA20')}, RSI(14)={signal.indicators.get('RSI14')}, RSI(2)={signal.indicators.get('RSI2')}

[최근 관련 뉴스 헤드라인]
{news_text}

[심의 절차 및 평가 원칙]
1. 🐂 [Bullish Researcher 관점]:
   - 기술적 과매도(RSI2, 볼린저 하단)에 따른 평균회귀 매수세, 지지선 유효성, 잠재 호재를 들어 반등 근거 제시.
2. 🐻 [Bearish Researcher 관점]:
   - 거시경제 역풍(금리, 인플레이션, 관세, 지정학)이나 실적/산업 악재로 인한 추가 급락 및 가짜 반등 위험을 경고.
3. ⚖️ [Chief Risk Officer (CRO) 최종 심의 및 Veto 권한]:
   - 매수 심사(BUY) 시:
     * 치명적/구조적 악재(-3 ~ -5점) 감지 시: 'BUY_VETO' (비중 0%, 매수 전면 거부 및 관망 권고) -> MDD 방어 최우선!
     * 불확실성/변동성 주의 요망: 'BUY_CAUTION' (비중 30~50% 축소 진입, 타이트한 손절)
     * 특이 악재 없는 통상적 조정(-2 ~ +1점): 'BUY_NORMAL' (비중 70%, 퀀트 승률 80% 신뢰 진입)
     * 강력 호재 동반(+2 ~ +5점): 'BUY_STRONG' (비중 100%, 강력 매수 승인)
   - 매도/보유 심사(SELL/HOLD) 시:
     * 보유 중 치명적 악재(-4 ~ -5점) 발생: 'EARLY_EXIT' (비중 100% 손절선 도달 전 선제 탈출)
     * 차트 익절 구간이나 강력 호재 지속(+3 ~ +5점): 'HOLD_EXTEND' (50%만 분할익절 후 본절 스탑 걸고 홀딩 연장)
     * 통상적 청산 시점: 'SELL_NOW' (비중 100% 계획대로 익절/손절 청산)

반드시 아래 JSON 형식으로만 응답해 주세요 (설명 없이 순수 JSON만 출력):
{{
  "sentiment_score": 0,
  "action_type": "BUY_NORMAL",
  "verdict": "🟡 매수 승인 (특이 악재 없음 - 권장 진입비중 70%)",
  "recommended_weight_pct": 70,
  "bull_argument": "RSI(2)=5.4 극단적 과매도 도달로 기술적 자율 반등 확률 매우 높음",
  "bear_argument": "중기 50일선 하회 상태로 매크로 지표 발표에 따른 변동성 주의 필요",
  "risk_assessment": "구조적 블랙스완 악재는 확인되지 않아 80% 퀀트 승률 기반 70% 비중 진입 승인",
  "news_summary_kr": "최근 영문 뉴스 핵심 요약 1~2줄 (한국어)",
  "swing_advice_kr": "차트와 리스크 심의를 결합한 구체적 매수/매도 실행 지침 1~2줄 (한국어)"
}}"""


def _extract_json(raw_text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(cleaned)


# OpenRouter 무료(:free) 모델 자동 우회 목록 (1순위 모델이 혼잡할 때 자동으로 다음 무료 모델 호출)
OPENROUTER_FREE_FALLBACKS = [
    "deepseek/deepseek-chat-v3-0324:free",
    "deepseek/deepseek-r1:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "openrouter/free",  # OpenRouter 공식 무료 모델 자동 라우터
]


def _call_openai_compatible(prompt: str) -> AINewsVerdict:
    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}",
        "HTTP-Referer": "https://github.com/etf-swing-bot",
        "X-Title": "US ETF Swing Signal Bot",
    }

    # OpenRouter 사용 시 기본 모델 + 무료 백업 모델 리스트 구성
    models_to_try = [LLM_MODEL]
    if "openrouter.ai" in LLM_BASE_URL:
        for fb in OPENROUTER_FREE_FALLBACKS:
            if fb not in models_to_try:
                models_to_try.append(fb)

    last_err = None
    for model_name in models_to_try:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert ETF swing trading assistant. Always respond in valid JSON format in Korean.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=35)
            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = _extract_json(content)
                return AINewsVerdict(**parsed)
            else:
                last_err = f"HTTP {resp.status_code}: {resp.text[:80]}"
        except Exception as e:
            last_err = str(e)
            continue

    raise RuntimeError(f"모든 무료 모델 호출 실패 ({last_err})")


def _call_gemini(prompt: str) -> AINewsVerdict:
    from google import genai

    client = genai.Client(api_key=GEMINI_API_KEY)
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt,
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": AINewsVerdict.model_json_schema(),
        },
    )
    return AINewsVerdict.model_validate_json(interaction.output_text)


def evaluate_signal_with_ai(
    signal: SwingSignalResult,
    news_items: list[dict],
    context_mode: str = "BUY",
) -> AINewsVerdict:
    """
    context_mode:
      - 'BUY' : 신규 매수 시그널에 대한 AI 뉴스 보조 판단 (Bull/Bear 토론 + 리스크 거부권 Veto 심사)
      - 'SELL': 매도 시그널 또는 보유 포지션에 대한 AI 뉴스 보조 판단 (전량청산 / 분할익절 후 홀딩연장 / 조기탈출)
    """
    prompt = _build_prompt(signal, news_items, context_mode)

    if AI_PROVIDER == "gemini":
        if not GEMINI_API_KEY or "your_" in GEMINI_API_KEY:
            return _smart_fallback_verdict(signal, news_items, context_mode, "GEMINI_API_KEY 미설정")
        try:
            return _call_gemini(prompt)
        except Exception as e:
            return _smart_fallback_verdict(signal, news_items, context_mode, f"Gemini 오류: {str(e)[:40]}")

    if AI_PROVIDER != "ollama" and (not LLM_API_KEY or "your_" in LLM_API_KEY):
        return _smart_fallback_verdict(signal, news_items, context_mode, f"{AI_PROVIDER.upper()} 키 미설정")

    try:
        return _call_openai_compatible(prompt)
    except Exception as e:
        return _smart_fallback_verdict(signal, news_items, context_mode, f"{AI_PROVIDER} 오류: {str(e)[:40]}")


def _smart_fallback_verdict(
    signal: Any,
    news_items: list[dict],
    context_mode: str,
    reason: str,
) -> AINewsVerdict:
    """API 키 연동 전에도 영문 헤드라인 키워드와 퀀트 지표로 Bull/Bear 토론 및 리스크 거부권 심사를 수행합니다."""
    score = _rule_based_news_score(news_items)
    headline = news_items[0]["title"] if news_items else "최근 수집된 주요 뉴스가 없습니다."
    rsi2_val = getattr(signal, "indicators", {}).get("RSI2", 10.0) if hasattr(signal, "indicators") else 10.0

    if context_mode == "BUY":
        if score <= -3:
            act = "BUY_VETO"
            verd = f"⛔ [매수 거부권 발동] 리스크 오프 권고 (악재 지수 [{score}점] / {reason})"
            weight = 0
            bull = "기술적 지표는 단기 과매도 구간이나, 펀더멘털 악재 앞에서는 지지선이 붕괴될 위험이 큼."
            bear = f"헤드라인에서 부정적 키워드([{score}점]) 다수 감지. '떨어지는 칼날' 추가 급락 위험 고조."
            risk = "치명적 악재로 판정하여 Chief Risk Officer 거부권(Veto) 행사 -> 매수 전면 취소 (MDD 방어)."
            adv = "차트상 과매도 타점이나 뉴스 악재가 확인되므로 진입을 보류하고 현금을 보존하세요."
        elif score >= 2:
            act = "BUY_STRONG"
            verd = f"🟢 강력 매수 승인 (호재 모멘텀 동반 [{score:+d}점] / 권장비중 100%)"
            weight = 100
            bull = f"RSI(2)={rsi2_val:.1f} 극단적 과매도 도달 + 우호적 뉴스 모멘텀으로 강력한 기술적 반등 탄력 기대."
            bear = "단기 갭상승 시 직전 매물대에서의 차익 실현 출회 가능성."
            risk = "거시적 악재 없음. 80.3% 퀀트 승률과 모멘텀 결합으로 100% 정석 비중 투입 승인."
            adv = "차트 타점과 긍정적 뉴스 분위기가 동반되었습니다. 정석 비중(100%)으로 스윙 진입하세요."
        else:
            act = "BUY_NORMAL"
            verd = f"🟡 일반 매수 승인 (특이 악재 없음 [{score:+d}점] / 권장비중 70%)"
            weight = 70
            bull = f"단기 과대낙폭(RSI2={rsi2_val:.1f})으로 80%+ 승률의 자율 반등 확률 지배적."
            bear = "시장의 전반적 관망세 및 50일선 아래 일시적 매물 소화 변동성 가능성."
            risk = "치명적 블랙스완 악재 부재. 70% 표준 비중 진입 승인 (5일선 회복 시 1차 익절 원칙 준수)."
            adv = f"심각한 악재 키워드가 없습니다. 퀀트 80% 승률에 기반해 진입하되 손절선을 준수하세요."
    else:
        if score >= 3:
            act = "HOLD_EXTEND"
            verd = f"🔥 분할 익절(50%) 후 홀딩 연장 권고 (호재 지속 [{score:+d}점])"
            weight = 50
            bull = "강력한 호재 모멘텀 지속으로 2차 목표가까지 추가 상승 룸 유효."
            bear = "차트상 단기 목표가 도달에 따른 차익 실현 매물 출회 가능성."
            risk = "원금 및 확정수익 보호를 위해 50% 분할 익절 후 손절선을 진입가(본절)로 상향 조정 권고."
            adv = "절반만 익절하여 수익을 확정하고, 나머지는 본절 스탑을 걸고 추세를 더 누리세요."
        elif score <= -4:
            act = "EARLY_EXIT"
            verd = f"🚨 조기 매도(선제 탈출) 경보 (악재 감지 [{score:+d}점])"
            weight = 100
            bull = "일시적 반등 기대 가능성 있으나 리스크 대비 보상 비율 급격히 악화."
            bear = f"심각한 뉴스 악재([{score}점]) 포착으로 지지선 붕괴 및 추가 급락 위험 고조."
            risk = "차트 손절선 도달 전이라도 선제적 리스크 오프(전량 청산) 권고."
            adv = "부정적 뉴스가 감지되었습니다. 추가 손실(MDD)을 막기 위해 신속히 포지션을 정리하세요."
        else:
            act = "SELL_NOW"
            verd = f"💰 계획대로 매도/익절 실행 (뉴스 중립 [{score:+d}점])"
            weight = 100
            bull = "기존 반등 목표치(5일선 회복 등) 달성 완료."
            bear = "목표가 부근 단기 매물 저항 및 상승 탄력 둔화."
            risk = "정석 퀀트 규칙에 따른 전량 청산 승인."
            adv = "차트 청산 원칙에 따라 정석대로 매도를 실행하여 수익을 확정하세요."

    return AINewsVerdict(
        sentiment_score=score,
        action_type=act,
        verdict=verd,
        recommended_weight_pct=weight,
        bull_argument=bull,
        bear_argument=bear,
        risk_assessment=risk,
        news_summary_kr=f"최신 헤드라인: {headline}",
        swing_advice_kr=adv,
    )
