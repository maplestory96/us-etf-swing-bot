"""
미국 ETF 스윙 시그널 봇 - 텔레그램(Telegram) 알림 발송 모듈
"""
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def send_telegram_message(message: str) -> bool:
    """
    텔레그램 봇 API를 통해 분석 리포트 메시지를 전송합니다.
    (토큰이나 챗 ID가 설정되지 않은 경우 안내 후 건너뜁니다.)
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or "your_" in TELEGRAM_BOT_TOKEN:
        print("\n[알림] .env 파일에 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID가 설정되지 않아 텔레그램 전송은 생략합니다.")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # 텔레그램 메시지 길이 제한(4096자) 대응 분할 전송
    max_len = 4000
    chunks = [message[i : i + max_len] for i in range(0, len(message), max_len)]

    success = True
    for chunk in chunks:
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": chunk,
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code != 200:
                print(f"[오류] 텔레그램 발송 실패 ({resp.status_code}): {resp.text}")
                success = False
        except Exception as e:
            print(f"[오류] 텔레그램 네트워크 오류: {e}")
            success = False

    if success:
        print("\n✅ 텔레그램으로 스윙 시그널 리포트 전송을 완료했습니다!")
    return success
