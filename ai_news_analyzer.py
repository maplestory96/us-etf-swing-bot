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
            "매수 시 ['BUY_STRONG'(강력매수 100%), 'BUY_NORMAL'(일반/분할매수), 'BUY_VETO'(악재로 매수보류)] / "
            "매도·보유 시 ['SELL_NOW'(전량청산/익절), 'HOLD_EXTEND'(호재동반 분할익절 후 홀딩연장), 'EARLY_EXIT'(악재로 조기탈출), 'HOLD_KEEP'(정상보유유지)]"
        )
    )
    verdict: str = Field(
        description="텔레그램에 표시할 한 줄 행동 지침 (예: '🟢 매수 승인 (호재 동반 - 비중 100%)', '⛔ 매수 보류 (악재 주의 - 관망 권장)', '🔥 절반 익절 후 홀딩 연장 (강력 호재)')"
    )
    recommended_weight_pct: int = Field(
        description="매수 시 권장 투입 비중 (0, 50, 70, 100 중 하나) 또는 매도 시 권장 매도 비중 (0, 50, 100 중 하나)"
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

    return f"""당신은 월스트리트 출신의 미국 주식 ETF 전문 퀀트+뉴스 스윙 트레이더입니다.
기본적으로 '차트(기술적 분석, 백테스트 승률 80.3%)'를 주된 매매 근거로 삼고, '뉴스/매크로 이슈'를 통해 매수·매도 결정을 최종 보조합니다.

[분석 모드]: {context_mode} ({'신규 매수 진입 심사' if context_mode == 'BUY' else '매도 타점 및 보유 포지션 심사'})
[포착된 ETF 정보]
- 종목: {signal.ticker} ({signal.name} / {signal.category})
- 현재가: ${signal.current_price} (전일 대비 {signal.change_pct:+.2f}%)
- 포착된 차트 시그널: {', '.join(active_tags)}
- 차트 근거: {' / '.join(active_reasons)}
- 주요 지표: 5일선=${signal.indicators.get('SMA5')}, 20일선=${signal.indicators.get('SMA20')}, RSI(14)={signal.indicators.get('RSI14')}, RSI(2)={signal.indicators.get('RSI2')}

[최근 관련 뉴스 헤드라인]
{news_text}

[의사결정 보조 규칙]
1. 매수 심사(BUY)일 때:
   - 뉴스가 단순 조정이나 중립(-2 ~ +1점)이면 차트 승률(80%)을 신뢰하여 'BUY_NORMAL' (권장비중 70%, 매수 승인)
   - 뚜렷한 호재나 금리/실적 모멘텀(+2 ~ +5점)이 있으면 'BUY_STRONG' (권장비중 100%, 강력 매수 승인)
   - 수출 통제, 관세 폭탄, 시스템 위기 등 심각한 구조적 악재(-3 ~ -5점)가 진행 중이면 차트 매수 타점이라도 'BUY_VETO' (권장비중 0%, 진입 보류 권고)
2. 매도/보유 심사(SELL/HOLD)일 때:
   - 차트 익절 타점이나 뉴스에 강력한 추가 상승 호재(+3 ~ +5점)가 터졌다면 'HOLD_EXTEND' (권장 매도비중 50% 분할익절 후 나머지는 홀딩 연장)
   - 일반적인 차트 익절/손절 타점이면 'SELL_NOW' (권장 매도비중 100% 계획대로 청산)
   - 보유 중 아직 손절선 전이라도 치명적 악재(-4 ~ -5점)가 발생했다면 'EARLY_EXIT' (권장 매도비중 100% 조기 탈출 경보)

반드시 아래 JSON 형식으로만 응답해 주세요 (설명 텍스트 없이 순수 JSON만 출력):
{{
  "sentiment_score": 0,
  "action_type": "BUY_NORMAL",
  "verdict": "🟡 매수 승인 (특이 악재 없음 - 권장 진입비중 70%)",
  "recommended_weight_pct": 70,
  "news_summary_kr": "최근 영문 뉴스 핵심 요약 1~2줄 (한국어)",
  "swing_advice_kr": "차트와 뉴스를 결합한 구체적 매수/매도 실행 지침 1~2줄 (한국어)"
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
      - 'BUY' : 신규 매수 시그널에 대한 AI 뉴스 보조 판단 (비중 결정 / 매수 거부권)
      - 'SELL': 매도 시그널 또는 보유 포지션에 대한 AI 뉴스 보조 판단 (전량청산 / 분할익절 후 홀딩연장 / 조기탈출)
    """
    prompt = _build_prompt(signal, news_items, context_mode)

    if AI_PROVIDER == "gemini":
        if not GEMINI_API_KEY or "your_" in GEMINI_API_KEY:
            return _smart_fallback_verdict(news_items, context_mode, "GEMINI_API_KEY 미설정")
        try:
            return _call_gemini(prompt)
        except Exception as e:
            return _smart_fallback_verdict(news_items, context_mode, f"Gemini 오류: {str(e)[:40]}")

    if AI_PROVIDER != "ollama" and (not LLM_API_KEY or "your_" in LLM_API_KEY):
        return _smart_fallback_verdict(news_items, context_mode, f"{AI_PROVIDER.upper()} 키 미설정")

    try:
        return _call_openai_compatible(prompt)
    except Exception as e:
        return _smart_fallback_verdict(news_items, context_mode, f"{AI_PROVIDER} 오류: {str(e)[:40]}")


def _smart_fallback_verdict(news_items: list[dict], context_mode: str, reason: str) -> AINewsVerdict:
    """API 키 연동 전에도 영문 헤드라인 키워드 분석으로 기본 매수/매도 보조 판단을 제공합니다."""
    score = _rule_based_news_score(news_items)
    headline = news_items[0]["title"] if news_items else "최근 수집된 주요 뉴스가 없습니다."

    if context_mode == "BUY":
        if score <= -3:
            act = "BUY_VETO"
            verd = f"⛔ 매수 보류 권고 (뉴스 악재 키워드 감지 [{score}점] / {reason})"
            weight = 0
            adv = "차트상 매수 타점이나 뉴스 헤드라인에 부정적 키워드가 다수 감지됩니다. 관망을 권장합니다."
        elif score >= 2:
            act = "BUY_STRONG"
            verd = f"🟢 강력 매수 승인 (뉴스 긍정 키워드 [{score:+d}점] / 권장비중 100%)"
            weight = 100
            adv = "차트 타점과 긍정적 뉴스 분위기가 동반되었습니다. 정석 비중(100%) 진입이 유리합니다."
        else:
            act = "BUY_NORMAL"
            verd = f"🟡 일반 매수 승인 (특이 악재 없음 [{score:+d}점] / 권장비중 70%)"
            weight = 70
            adv = f"심각한 악재 키워드가 없습니다. 백테스트 승률(80.3%)에 기반해 진입하되 손절선을 준수하세요. (.env에 AI 키 입력 시 심층 분석 활성화)"
    else:
        if score >= 3:
            act = "HOLD_EXTEND"
            verd = f"🔥 분할 익절(50%) 후 홀딩 연장 권고 (호재 감지 [{score:+d}점])"
            weight = 50
            adv = "차트상 매도 구간이나 긍정적 뉴스 모멘텀이 강합니다. 절반만 익절하고 나머지는 추세를 더 타세요."
        elif score <= -4:
            act = "EARLY_EXIT"
            verd = f"🚨 조기 매도(선제 탈출) 경보 (악재 감지 [{score:+d}점])"
            weight = 100
            adv = "부정적 뉴스가 감지되었습니다. 리스크 관리를 위해 신속히 포지션을 정리하세요."
        else:
            act = "SELL_NOW"
            verd = f"💰 계획대로 매도/익절 실행 (뉴스 중립 [{score:+d}점])"
            weight = 100
            adv = "차트 청산 원칙(5일선 회복 익절 또는 손절선 기준)에 따라 정석대로 매도를 실행하세요."

    return AINewsVerdict(
        sentiment_score=score,
        action_type=act,
        verdict=verd,
        recommended_weight_pct=weight,
        news_summary_kr=f"최신 헤드라인: {headline}",
        swing_advice_kr=adv,
    )
