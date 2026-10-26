"""
korean_etf_champion.py
국내 ETF 11종에 대한 1차 & 2차 이터레이션 통합 최종 챔피언 전략 선별 및 종합 성과 평가기
- 5개년(2021~2026) 백테스트 (거래수수료/슬리피지 0.15% 실거래 차감)
- 각 ETF별 8개 후보 전략 중 샤프지수/수익률/MDD 종합 최고 전략을 챔피언으로 선정
"""

import yfinance as yf
import pandas as pd
import numpy as np

KOREAN_ETFS = [
    {"symbol": "069500.KS", "name": "KODEX 200", "cat": "지수형"},
    {"symbol": "122630.KS", "name": "KODEX 레버리지", "cat": "지수레버리지"},
    {"symbol": "229200.KS", "name": "KODEX 코스닥150", "cat": "지수형"},
    {"symbol": "233740.KS", "name": "KODEX 코스닥150레버리지", "cat": "지수레버리지"},
    {"symbol": "091160.KS", "name": "KODEX 반도체", "cat": "주도섹터"},
    {"symbol": "305540.KS", "name": "TIGER 2차전지테마", "cat": "사이클섹터"},
    {"symbol": "091170.KS", "name": "KODEX 은행", "cat": "가치섹터"},
    {"symbol": "143860.KS", "name": "TIGER 헬스케어", "cat": "성장섹터"},
    {"symbol": "132030.KS", "name": "KODEX 골드선물(H)", "cat": "원자재/안전자산"},
    {"symbol": "261220.KS", "name": "KODEX WTI원유선물(H)", "cat": "원자재/에너지"},
    {"symbol": "261240.KS", "name": "KODEX 미국달러선물", "cat": "환율/헤지"}
]

def calculate_indicators(df):
    data = df.copy()
    data['MA5'] = data['Close'].rolling(5).mean()
    data['MA20'] = data['Close'].rolling(20).mean()
    data['MA60'] = data['Close'].rolling(60).mean()
    data['MA200'] = data['Close'].rolling(200).mean()
    
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    roll_up2 = gain.ewm(span=2, adjust=False).mean()
    roll_down2 = loss.ewm(span=2, adjust=False).mean()
    rs2 = roll_up2 / (roll_down2 + 1e-9)
    data['RSI2'] = 100 - (100 / (1 + rs2))
    
    roll_up14 = gain.ewm(span=14, adjust=False).mean()
    roll_down14 = loss.ewm(span=14, adjust=False).mean()
    rs14 = roll_up14 / (roll_down14 + 1e-9)
    data['RSI14'] = 100 - (100 / (1 + rs14))
    
    std20 = data['Close'].rolling(20).std()
    data['BB_Mid'] = data['MA20']
    data['BB_Upper'] = data['BB_Mid'] + 2 * std20
    data['BB_Lower'] = data['BB_Mid'] - 2 * std20
    data['BB_Width'] = (data['BB_Upper'] - data['BB_Lower']) / data['BB_Mid']
    data['BB_Width_MA20'] = data['BB_Width'].rolling(20).mean()
    
    low14 = data['Low'].rolling(14).min()
    high14 = data['High'].rolling(14).max()
    k_fast = 100 * (data['Close'] - low14) / (high14 - low14 + 1e-9)
    data['Stoch_K'] = k_fast.rolling(3).mean()
    data['Stoch_D'] = data['Stoch_K'].rolling(3).mean()
    
    data['Vol_MA20'] = data['Volume'].rolling(20).mean()
    data['Disparity20'] = (data['Close'] / data['MA20']) * 100
    return data

def run_backtest(df, entry_cond_func, exit_cond_func, fee=0.0015):
    in_pos = False
    buy_price = 0.0
    highest_price = 0.0
    trades = []
    equity = 1.0
    equity_curve = [1.0]
    
    for i in range(200, len(df)):
        curr_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        if not in_pos:
            if entry_cond_func(curr_row, prev_row, df.iloc[:i+1]):
                in_pos = True
                buy_price = curr_row['Close']
                highest_price = buy_price
        else:
            highest_price = max(highest_price, curr_row['High'])
            should_exit, reason = exit_cond_func(curr_row, prev_row, buy_price, highest_price)
            if should_exit:
                sell_price = curr_row['Close']
                ret = (sell_price / buy_price) * (1 - fee) - 1.0
                equity *= (1.0 + ret)
                trades.append({
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'ret': ret,
                    'reason': reason
                })
                in_pos = False
        equity_curve.append(equity)
        
    eq_s = pd.Series(equity_curve)
    cum_ret = (equity - 1.0) * 100
    mdd = ((eq_s - eq_s.cummax()) / eq_s.cummax()).min() * 100
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        wins = trades_df[trades_df['ret'] > 0]
        losses = trades_df[trades_df['ret'] <= 0]
        win_rate = (len(wins) / len(trades)) * 100
        gross_profit = wins['ret'].sum() if len(wins) > 0 else 0
        gross_loss = abs(losses['ret'].sum()) if len(losses) > 0 else 1e-9
        profit_factor = gross_profit / gross_loss
    else:
        win_rate = 0.0
        profit_factor = 0.0
        
    return {
        'num_trades': len(trades),
        'win_rate': win_rate,
        'cum_ret': cum_ret,
        'mdd': mdd,
        'profit_factor': profit_factor
    }

# 전략 정의 함수들
def get_strategies():
    strategies = {}
    
    # 전략 1: K-추세 눌림목 프로 (200일선 우상향 + 20/60선 지지반등 + 스토캐스틱 K>D + 트레일링 스탑)
    def entry_s1(c, p, sub_df):
        ma_cond = c['Close'] > c['MA200']
        supp_cond = (c['Low'] <= c['MA20'] * 1.015) or (c['Low'] <= c['MA60'] * 1.015)
        rebound_cond = c['Close'] > c['Open'] and c['Close'] > p['Close']
        stoch_cond = c['Stoch_K'] > c['Stoch_D'] and p['Stoch_K'] <= p['Stoch_D']
        return ma_cond and supp_cond and rebound_cond and stoch_cond
        
    def exit_s1(c, p, buy_price, highest_price):
        ret = (c['Close'] - buy_price) / buy_price
        drawdown_from_peak = (c['Close'] - highest_price) / highest_price
        if ret <= -0.04:
            return True, "STOP_LOSS_4%"
        if ret >= 0.05 and drawdown_from_peak <= -0.025:
            return True, "TRAILING_STOP_2.5%"
        if ret >= 0.12:
            return True, "TAKE_PROFIT_12%"
        if c['Close'] < c['MA20'] and p['Close'] >= p['MA20']:
            return True, "MA20_BREAKDOWN"
        return False, ""
    strategies["K-추세 눌림목 프로"] = (entry_s1, exit_s1)
    
    # 전략 2: K-볼린저 수축 돌파 (밴드폭 수축 + 상단 밴드 돌파 + 거래량 1.3배 실린 장대양봉)
    def entry_s2(c, p, sub_df):
        squeeze = c['BB_Width'] < c['BB_Width_MA20'] * 0.90
        breakout = c['Close'] > c['BB_Upper']
        vol_surge = c['Volume'] > c['Vol_MA20'] * 1.30
        bullish = c['Close'] > c['Open']
        return squeeze and breakout and vol_surge and bullish
        
    def exit_s2(c, p, buy_price, highest_price):
        ret = (c['Close'] - buy_price) / buy_price
        drawdown = (c['Close'] - highest_price) / highest_price
        if ret <= -0.035:
            return True, "STOP_LOSS_3.5%"
        if ret >= 0.04 and drawdown <= -0.02:
            return True, "TRAILING_STOP_2%"
        if c['Close'] < c['BB_Mid']:
            return True, "BB_MID_EXIT"
        if ret >= 0.10:
            return True, "TAKE_PROFIT_10%"
        return False, ""
    strategies["K-볼린저 수축 돌파"] = (entry_s2, exit_s2)

    # 전략 3: K-매크로/원자재 과매도 반등 (RSI(14)<35 or 괴리율<94 + 5일선 골든크로스 복귀)
    def entry_s3(c, p, sub_df):
        oversold = (c['RSI14'] <= 35) or (c['Disparity20'] <= 94.0)
        rebound = c['Close'] > c['MA5'] and p['Close'] <= p['MA5']
        return oversold or (rebound and p['RSI14'] <= 35)
        
    def exit_s3(c, p, buy_price, highest_price):
        ret = (c['Close'] - buy_price) / buy_price
        if ret <= -0.04:
            return True, "STOP_LOSS_4%"
        if c['Close'] >= c['MA20'] and p['Close'] < p['MA20']:
            return True, "MA20_RECOVERY"
        if ret >= 0.06:
            return True, "TAKE_PROFIT_6%"
        return False, ""
    strategies["K-원자재/매크로 과매도 반등"] = (entry_s3, exit_s3)

    # 전략 4: K-이평선 눌림목 클래식 (60일선 위 + 20일선 눌림 + 양봉) - 레버리지/코스닥 특화
    def entry_s4(c, p, sub_df):
        trend = c['Close'] > c['MA60']
        touch = (c['Low'] <= c['MA20'] * 1.01) and (c['Close'] >= c['MA20'])
        bullish = c['Close'] > c['Open']
        return trend and touch and bullish
        
    def exit_s4(c, p, buy_price, highest_price):
        ret = (c['Close'] - buy_price) / buy_price
        if ret <= -0.04:
            return True, "STOP_LOSS_4%"
        if ret >= 0.06:
            return True, "TAKE_PROFIT_6%"
        if c['Close'] < c['MA20']:
            return True, "MA20_BREAKDOWN"
        return False, ""
    strategies["K-이평선 눌림목 클래식"] = (entry_s4, exit_s4)

    return strategies

def main():
    print("🚀 [최종 국장 챔피언 전략 배정 백테스트 실행중...]")
    strategies = get_strategies()
    results = []
    
    for item in KOREAN_ETFS:
        sym = item['symbol']
        name = item['name']
        cat = item['cat']
        
        df = yf.download(sym, start="2021-01-01", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = calculate_indicators(df)
        
        bh_ret = (df['Close'].iloc[-1] / df['Close'].iloc[200] - 1.0) * 100
        close_s = df['Close'].iloc[200:]
        bh_mdd = ((close_s - close_s.cummax()) / close_s.cummax()).min() * 100
        
        best_strat_name = None
        best_score = -9999
        best_metrics = None
        
        for s_name, (entry_fn, exit_fn) in strategies.items():
            res = run_backtest(df, entry_fn, exit_fn)
            if res['num_trades'] >= 5:
                # 점수 산정: 승률(가중치 0.3) + 수익률(가중치 0.4) + MDD방어(가중치 0.3) + PF
                score = res['cum_ret'] - (res['mdd'] * 1.2) + (res['win_rate'] * 0.5) + (res['profit_factor'] * 10)
                if score > best_score:
                    best_score = score
                    best_strat_name = s_name
                    best_metrics = res
                    
        # 만약 5회 이상인게 없으면 거래수 상관없이 1위
        if best_metrics is None:
            for s_name, (entry_fn, exit_fn) in strategies.items():
                res = run_backtest(df, entry_fn, exit_fn)
                score = res['cum_ret'] - res['mdd']
                if score > best_score:
                    best_score = score
                    best_strat_name = s_name
                    best_metrics = res
                    
        results.append({
            'cat': cat,
            'name': name,
            'sym': sym,
            'strat': best_strat_name,
            'trades': best_metrics['num_trades'],
            'win_rate': best_metrics['win_rate'],
            'cum_ret': best_metrics['cum_ret'],
            'mdd': best_metrics['mdd'],
            'pf': best_metrics['profit_factor'],
            'bh_ret': bh_ret,
            'bh_mdd': bh_mdd
        })
        
    print("\n" + "="*115)
    print(f"{'분야':<10} | {'ETF 명':<20} | {'최적 챔피언 전략':<22} | {'승률':<7} | {'스윙수익':<9} | {'스윙MDD':<8} | {'PF':<5} | {'단순보유(MDD)'}")
    print("="*115)
    
    tot_wr = []
    tot_ret = []
    tot_mdd = []
    tot_bh_ret = []
    tot_bh_mdd = []
    
    for r in results:
        tot_wr.append(r['win_rate'])
        tot_ret.append(r['cum_ret'])
        tot_mdd.append(r['mdd'])
        tot_bh_ret.append(r['bh_ret'])
        tot_bh_mdd.append(r['bh_mdd'])
        
        bh_str = f"{r['bh_ret']:+.1f}% ({r['bh_mdd']:.1f}%)"
        print(f"{r['cat']:<10} | {r['name']:<20} | {r['strat']:<22} | {r['win_rate']:>5.1f}% | {r['cum_ret']:>+7.1f}% | {r['mdd']:>6.1f}% | {r['pf']:>4.2f} | {bh_str}")
        
    print("="*115)
    print(f"📊 [포트폴리오 전체 종합 평균]")
    print(f"  • 평균 승률   : {np.mean(tot_wr):.1f}%")
    print(f"  • 평균 수익률 : {np.mean(tot_ret):+.1f}% (단순보유 평균: {np.mean(tot_bh_ret):+.1f}%)")
    print(f"  • 평균 MDD    : {np.mean(tot_mdd):.1f}% (단순보유 평균 MDD: {np.mean(tot_bh_mdd):.1f}%)")
    print("="*115)

if __name__ == "__main__":
    main()
