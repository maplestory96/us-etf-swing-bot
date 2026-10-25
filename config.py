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
