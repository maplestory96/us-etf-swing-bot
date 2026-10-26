"""
미국 ETF 스윙 시그널 봇 - 텔레그램(Telegram) 알림 발송 모듈
"""
import requests
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID


def _diagnose_bot() -> None:
    """발송에 사용 중인 봇 계정과 대상 대화방 정보를 로그에 출력합니다 (토큰 값은 출력하지 않음)."""
    base = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    try:
        me = requests.get(f"{base}/getMe", timeout=10).json()
        if me.get("ok"):
            print(f"[진단] 발송 봇 계정: @{me['result'].get('username')} ({me['result'].get('first_name')})")
        else:
            print(f"[진단] 봇 토큰 오류: {me}")
    except Exception as e:
        print(f"[진단] getMe 실패: {e}")

    try:
        chat = requests.get(f"{base}/getChat", params={"chat_id": TELEGRAM_CHAT_ID}, timeout=10).json()
        if chat.get("ok"):
            r = chat["result"]
            name = r.get("title") or f"{r.get('first_name', '')} {r.get('last_name', '')}".strip()
            print(f"[진단] 대상 대화방: type={r.get('type')} / 이름={name} / username=@{r.get('username')}")
        else:
            print(f"[진단] Chat ID 오류: {chat}")
    except Exception as e:
        print(f"[진단] getChat 실패: {e}")


def send_telegram_message(message: str) -> bool:
    """
    텔레그램 봇 API를 통해 분석 리포트 메시지를 전송합니다.
    (토큰이나 챗 ID가 설정되지 않은 경우 안내 후 건너뜁니다.)
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID or "your_" in TELEGRAM_BOT_TOKEN:
        print("\n[알림] .env 파일에 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID가 설정되지 않아 텔레그램 전송은 생략합니다.")
        return False

    _diagnose_bot()

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

    # 텔레그램 메시지 길이 제한(4096자) 대응 분할 전송
    max_len = 4000
    chunks = [message[i : i + max_len] for i in range(0, len(message), max_len)]

    success = True
    for idx, chunk in enumerate(chunks, 1):
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
            else:
                msg_id = resp.json().get("result", {}).get("message_id")
                print(f"[진단] {idx}/{len(chunks)}번째 메시지 전송 OK (message_id={msg_id})")
        except Exception as e:
            print(f"[오류] 텔레그램 네트워크 오류: {e}")
            success = False

    if success:
        print("\n✅ 텔레그램으로 스윙 시그널 리포트 전송을 완료했습니다!")
    return success
