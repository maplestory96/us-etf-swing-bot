import warnings
import xml.etree.ElementTree as ET
from urllib.parse import quote
import pandas as pd
import requests
import yfinance as yf

# yfinance 내부의 pandas 향후 버전 경고 숨김 처리
warnings.filterwarnings("ignore", category=Warning)


def fetch_ohlcv(ticker: str, period: str = "1y") -> pd.DataFrame:
    """
    야후 파이낸스(yfinance)를 통해 일봉 OHLCV 데이터를 가져옵니다.
    auto_adjust=False로 설정하여 배당금 재투자 왜곡이 없는 실제 차트 종가(Raw Price)를 사용합니다.
    """
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d", auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        # 컬럼 표준화
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
        return df
    except Exception as e:
        print(f"[오류] {ticker} 차트 데이터 수집 실패: {e}")
        return pd.DataFrame()


def fetch_etf_news(ticker: str, etf_name: str = "", max_items: int = 5) -> list[dict]:
    """
    1순위: yfinance 내장 뉴스 데이터 수집
    2순위(보완): Google News RSS 피드를 통해 해당 ETF 및 섹터 최신 뉴스 수집
    """
    news_list = []

    # 1. yfinance 내장 뉴스 시도
    try:
        stock = yf.Ticker(ticker)
        yf_news = stock.news
        if yf_news:
            for item in yf_news[:max_items]:
                # 최신 yfinance 구조 대응 (content 딕셔너리 내부 또는 최상위 키)
                content = item.get("content", item)
                title = content.get("title", "")
                summary = content.get("summary", content.get("description", ""))
                provider = content.get("provider", {})
                publisher = provider.get("displayName", item.get("publisher", "Yahoo Finance")) if isinstance(provider, dict) else str(provider)

                if title:
                    news_list.append({
                        "title": title,
                        "summary": summary[:200] if summary else "",
                        "publisher": publisher,
                    })
    except Exception:
        pass

    # 2. 뉴스가 부족할 경우 Google News RSS에서 추가 수집
    if len(news_list) < 3:
        try:
            query = quote(f"{ticker} ETF stock market")
            rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
            resp = requests.get(rss_url, timeout=5)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:max_items]:
                    title = item.findtext("title", "")
                    source = item.findtext("source", "Google News")
                    if title and not any(n["title"] == title for n in news_list):
                        news_list.append({
                            "title": title,
                            "summary": "",
                            "publisher": source,
                        })
                        if len(news_list) >= max_items:
                            break
        except Exception as e:
            print(f"[경고] {ticker} RSS 뉴스 수집 중 오류: {e}")

    return news_list[:max_items]
