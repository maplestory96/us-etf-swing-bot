"""
미국 ETF 스윙 시그널 봇 - 설정 및 감시 종목(유니버스) 관리 모듈
5년 백테스팅 결과를 반영하여 ETF 성격별로 최적의 스윙 전략을 자동 배정(Smart Routing)합니다.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API 및 텔레그램 환경변수 로드 (기본값: 결제 필요 없는 OpenRouter 무료 등급)
AI_PROVIDER = os.getenv("AI_PROVIDER", "openrouter").strip().lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1").strip().rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek/deepseek-chat-v3-0324:free").strip()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()

# 감시할 미국 주요 ETF 유니버스 및 종목별 맞춤 전략 모드(mode)
# - "OVERSOLD_ONLY": 주가지수/반도체/3배레버리지 -> 승률 80%+ 검증된 [💎과매도 반등] 전용 스나이핑 (가짜 고점돌파 배제)
# - "MACRO_CYCLE"  : 원자재/에너지/채권/금융 -> [🔥거래량 폭발 박스권 돌파] 주력 + [💎극단적 과매도 반등] 병행
ETF_UNIVERSE = {
    # 1. 대표 시장 지수 (1배수) -> 과매도 반등 스나이핑 전용 (QQQ 승률 84.1%, MDD -8.3%)
    "SPY": {"name": "S&P 500 지수 ETF", "category": "시장지수(1x)", "mode": "OVERSOLD_ONLY"},
    "QQQ": {"name": "나스닥 100 지수 ETF", "category": "시장지수(1x)", "mode": "OVERSOLD_ONLY"},
    "DIA": {"name": "다우존스 30 지수 ETF", "category": "시장지수(1x)", "mode": "OVERSOLD_ONLY"},
    "IWM": {"name": "러셀 2000 중소형주 ETF", "category": "시장지수(1x)", "mode": "OVERSOLD_ONLY"},

    # 2. 기술/성장/반도체 섹터 (1배수) -> 과매도 반등 스나이핑 전용 (SOXX 승률 82.8%, 수익 +96.7%)
    "SOXX": {"name": "필라델피아 반도체 ETF", "category": "반도체/기술(1x)", "mode": "OVERSOLD_ONLY"},
    "SMH": {"name": "반에크 반도체 ETF", "category": "반도체/기술(1x)", "mode": "OVERSOLD_ONLY"},
    "XLK": {"name": "기술주 섹터 ETF", "category": "반도체/기술(1x)", "mode": "OVERSOLD_ONLY"},
    "ARKK": {"name": "아크 혁신 테마 ETF", "category": "반도체/기술(1x)", "mode": "OVERSOLD_ONLY"},

    # 3. 고수익 스윙용 레버리지 ETF (3배수) -> 과매도 반등 스나이핑 전용 (SOXL +183%, TQQQ +178%, 승률 80%+)
    "TQQQ": {"name": "나스닥 100 3배 레버리지 ETF", "category": "레버리지(3x)", "mode": "OVERSOLD_ONLY"},
    "SOXL": {"name": "반도체 3배 레버리지 ETF", "category": "레버리지(3x)", "mode": "OVERSOLD_ONLY"},
    "UPRO": {"name": "S&P 500 3배 레버리지 ETF", "category": "레버리지(3x)", "mode": "OVERSOLD_ONLY"},

    # 4. 사이클 섹터 및 매크로 (원자재/에너지/채권/크립토) -> 박스권 돌파 + 과매도 병행 (MACRO_CYCLE)
    "XLE": {"name": "에너지 섹터 ETF", "category": "사이클/매크로", "mode": "MACRO_CYCLE"},
    "GLD": {"name": "금 현물 ETF", "category": "사이클/매크로", "mode": "MACRO_CYCLE"},
    "USO": {"name": "WTI 원유 선물 ETF", "category": "사이클/매크로", "mode": "MACRO_CYCLE"},
    "TLT": {"name": "미국 20년 이상 장기국채 ETF", "category": "사이클/매크로", "mode": "MACRO_CYCLE"},
    "TMF": {"name": "미국 장기국채 3배 레버리지 ETF", "category": "사이클/매크로(3x)", "mode": "MACRO_CYCLE"},
    "IBIT": {"name": "현물 비트코인 ETF", "category": "크립토/대체자산", "mode": "MACRO_CYCLE"},
    "XLF": {"name": "금융주 섹터 ETF", "category": "전통섹터(1x)", "mode": "MACRO_CYCLE"},
    "XLV": {"name": "헬스케어 섹터 ETF", "category": "전통섹터(1x)", "mode": "MACRO_CYCLE"},
    "XLI": {"name": "산업재 섹터 ETF", "category": "전통섹터(1x)", "mode": "MACRO_CYCLE"},
    "XLY": {"name": "임의소비재 섹터 ETF", "category": "전통섹터(1x)", "mode": "MACRO_CYCLE"},
}

# -------------------------------------------------------------------------
# 국내(한국장) 대표 ETF 11종 유니버스 및 5년 백테스트 검증 챔피언 전략 배정
# -------------------------------------------------------------------------
# - KR_PULLBACK_PRO   : [K-추세 눌림목 프로] 200일선 우상향 필터 + 20/60선 지지반등 + 스토캐스틱 K>D + 트레일링스탑
# - KR_BB_SQUEEZE     : [K-볼린저 수축 돌파] 밴드폭 극도 수축(스퀴즈) 후 거래량 1.3배 폭발 상단 돌파
# - KR_OVERSOLD_REBOUND: [K-원자재/매크로 과매도 반등] RSI(14) 35이하/괴리율 침체 후 5일선 양봉 회복
# - KR_PULLBACK_CLASSIC: [K-이평선 눌림목 클래식] 60일선 위 20일선 눌림목 반등 (코스닥 레버리지 특화)
KOREAN_ETF_UNIVERSE = {
    # 1. 지수 및 레버리지 (200일선 필터 추세 눌림목 챔피언)
    "069500.KS": {"name": "KODEX 200", "category": "국내지수(1x)", "strategy": "KR_PULLBACK_PRO", "memo": "MDD -6.9% 극강 방어"},
    "122630.KS": {"name": "KODEX 레버리지", "category": "국내지수(2x)", "strategy": "KR_PULLBACK_PRO", "memo": "5년 +136.1% (MDD -13%)"},
    "229200.KS": {"name": "KODEX 코스닥150", "category": "코스닥지수(1x)", "strategy": "KR_PULLBACK_PRO", "memo": "승률 64.3% (MDD -8.3%)"},
    "233740.KS": {"name": "KODEX 코스닥150레버리지", "category": "코스닥지수(2x)", "strategy": "KR_PULLBACK_CLASSIC", "memo": "단순보유 폭락 회피"},

    # 2. 주도 & 사이클 섹터 (반도체/2차전지/은행/헬스케어)
    "091160.KS": {"name": "KODEX 반도체", "category": "주도섹터", "strategy": "KR_PULLBACK_PRO", "memo": "5년 +71.2% (MDD -10.8%)"},
    "305540.KS": {"name": "TIGER 2차전지테마", "category": "사이클섹터", "strategy": "KR_BB_SQUEEZE", "memo": "승률 80.0% / PF 7.64 (MDD -5.6%)"},
    "091170.KS": {"name": "KODEX 은행", "category": "가치섹터", "strategy": "KR_OVERSOLD_REBOUND", "memo": "승률 63.0% 과매도 스나이핑"},
    "143860.KS": {"name": "TIGER 헬스케어", "category": "성장섹터", "strategy": "KR_OVERSOLD_REBOUND", "memo": "승률 61.0% 반등 스나이핑"},

    # 3. 원자재 및 환율/헤지 (골드/원유/달러)
    "132030.KS": {"name": "KODEX 골드선물(H)", "category": "원자재/안전자산", "strategy": "KR_BB_SQUEEZE", "memo": "승률 75.0% / PF 5.98 (MDD -3.0%)"},
    "261220.KS": {"name": "KODEX WTI원유선물(H)", "category": "원자재/에너지", "strategy": "KR_OVERSOLD_REBOUND", "memo": "5년 +86.7% / 승률 63.3%"},
    "261240.KS": {"name": "KODEX 미국달러선물", "category": "환율/헤지", "strategy": "KR_BB_SQUEEZE", "memo": "승률 62.5% / MDD -2.3%"},
}

# 스윙 전략 기본 파라미터
STRATEGY_PARAMS = {
    # 1. 과매도 반등 (Oversold Bounce)
    "oversold_rsi14_threshold": 35.0, # RSI(14) 과매도 기준
    "oversold_rsi2_threshold": 15.0,  # 단기 RSI(2) 극단적 과매도 기준 (래리 코너스 전략)

    # 2. 박스권 돌파 (Breakout - MACRO_CYCLE 종목 전용)
    "breakout_lookback_days": 20,     # 최근 20일 신고가 돌파 기준
    "breakout_volume_ratio": 1.35,    # 20일 평균 거래량 대비 135% 이상 폭발 필수

    # 3. 손익비 설정 (ATR 기반 스윙 손절/익절 배수)
    "atr_stop_loss_multiplier": 2.2,  # 진입가 - (2.2 * ATR)
    "atr_target_1_multiplier": 2.5,   # 1차 목표가: 진입가 + (2.5 * ATR)
    "atr_target_2_multiplier": 4.0,   # 2차 목표가: 진입가 + (4.0 * ATR)
}

