import json, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import requests, yfinance as yf

ASOF=pd.Timestamp('2026-09-01'); START=1996; CONTRIB=1000.0
SCHEDULES={'Jan 1':(1,1),'Mar 1':(3,1),'Jun 1':(6,1),'Sep 1':(9,1)}
URL='https://raw.githubusercontent.com/fja05680/sp500/master/sp500_ticker_start_end.csv'
ALIASES={'FB':'META','ANTM':'ELV','WLTW':'WTW','PCLN':'BKNG','VIAC':'PARA','CTL':'LUMN','JEC':'J','LB':'BBWI','NLOK':'GEN','PKI':'RVTY','FISV':'FI','ABC':'COR','RE':'EG','HFC':'DINO','BK':'BNY'}
OUT=Path('outputs'); OUT.mkdir(exist_ok=True)

def norm(t): return ALIASES.get(str(t).strip().upper(),str(t).strip().upper()).replace('.','-')
def get_members():
    r=requests.get(URL,timeout=60); r.raise_for_status(); Path('membership.csv').write_bytes(r.content)
    m=pd.read_csv('membership.csv',dtype=str); m.ticker=m.ticker.str.upper().str.strip(); m.start_date=pd.to_datetime(m.start_date); m.end_date=pd.to_datetime(m.end_date).fillna(pd.Timestamp('2100-01-01')); m['symbol']=m.ticker.map(norm); return m

def dl(symbols,start,end):
    symbols=sorted(set(symbols)); pieces=[]
    for i in range(0,len(symbols),80):
        b=symbols[i:i+80]; print('download',i,'/',len(symbols),flush=True)
        raw=yf.download(b,start=start.strftime('%Y-%m-%d'),end=(end+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),auto_adjust=False,actions=False,group_by='ticker',threads=True,progress=False,timeout=60)
        if raw.empty: continue
        for s in b:
            try:
                x=raw[s].copy() if isinstance(raw.columns,pd.MultiIndex) and s in raw.columns.get_level_values(0) else raw.xs(s,level=1,axis=1).copy()
            except: continue
            x.columns=[str(c).lower().replace(' ','_') for c in x.columns]
            if 'adj_close' not in x: x['adj_close']=x.get('close')
            if 'open' not in x or 'close' not in x: continue
            x['adj_open']=np.where(x['close'].abs()>1e-12,x['open']*x['adj_close']/x['close'],np.nan)
            x=x[['adj_open','adj_close']].dropna(how='all'); x['symbol']=s; x['date']=pd.to_datetime(x.index).tz_localize(None).normalize(); pieces.append(x.reset_index(drop=True))
        time.sleep(.3)
    return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=['adj_open','adj_close','symbol','date'])

def one(symbol,start,end):
    for _ in range(3):
        try:
            x=yf.download(symbol,start=start.strftime('%Y-%m-%d'),end=(end+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),auto_adjust=False,actions=False,progress=False,threads=False,timeout=40)
            if not x.empty:
                if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
                x.columns=[str(c).lower().replace(' ','_') for c in x.columns]; x['adj_close']=x.get('adj_close',x.get('close')); x['adj_open']=np.where(x['close'].abs()>1e-12,x['open']*x['adj_close']/x['close'],np.nan); x.index=pd.to_datetime(x.index).tz_localize(None).normalize(); return x[['adj_open','adj_close']]
        except: pass
        time.sleep(2)
    return pd.DataFrame()

def trade_dates(spy):
    dates=spy.index.sort_values(); ev=[]
    for name,(mo,da) in SCHEDULES.items():
        for y in range(START,ASOF.year+1):
            cal=pd.Timestamp(y,mo,da); future=dates[(dates>=cal)&(dates<=ASOF)]
            if len(future)==0: continue
            td=future[0]; pos=dates.get_loc(td)
            if pos==0: continue
            sig=dates[pos-1]; target=sig-pd.DateOffset(years=1); past=dates[dates<=target]
            if len(past): ev.append((name,y,cal,td,sig,past[-1]))
    return ev

def rank_event(panel,m,ev):
    name,y,cal,td,sig,lb=ev; eligible=m[(m.start_date<=td)&(m.end_date>td)][['ticker','symbol']].drop_duplicates('ticker')
    rows=[]
    for r in eligible.itertuples(index=False):
        x=panel[panel.symbol.eq(r.symbol)]
        if x.empty: continue
        a=x[(x.date<=lb)&(x.date>=lb-pd.Timedelta(days=10))]; b=x[(x.date<=sig)&(x.date>=sig-pd.Timedelta(days=10))]; c=x[(x.date>=td)&(x.date<=td+pd.Timedelta(days=7))]; w=x[(x.date>lb)&(x.date<=sig)].sort_values('date')
        if a.empty or b.empty or c.empty or len(w)<200: continue
        pa=float(a.sort_values('date').iloc[-1].adj_close); pb=float(b.sort_values('date').iloc[-1].adj_close); po=float(c.sort_values('date').iloc[0].adj_open)
        if not (pa>0 and pb>0 and po>0): continue
        dr=w.adj_close.pct_change()
        if (dr.max(skipna=True)>2.0) or (dr.min(skipna=True)<-.8): continue
        rows.append((r.ticker,r.symbol,pb/pa-1,len(w),po))
    if not rows: raise RuntimeError(f'no rank {name} {y}')
    q=pd.DataFrame(rows,columns=['ticker','symbol','signal_return','obs','buy_px']).sort_values('signal_return',ascending=False); win=q.iloc[0]
    return {'schedule':name,'year':y,'calendar_date':cal,'trade_date':td,'signal_date':sig,'lookback_date':lb,'eligible':len(eligible),'ticker':win.ticker,'symbol':win.symbol,'signal_return':win.signal_return,'obs':int(win.obs)}

def growth(panel,ticker,symbol,start,end,final=False):
    x=panel[panel.symbol.eq(symbol)].sort_values('date'); buy=x[(x.date>=start)&(x.date<=start+pd.Timedelta(days=7))]
    if buy.empty:
        z=one(norm(ticker),start-pd.Timedelta(days=5),end+pd.Timedelta(days=5))
        if z.empty: raise RuntimeError(f'no holding {ticker} {start}')
        bp=z[(z.index>=start)&(z.index<=start+pd.Timedelta(days=7))]
        if bp.empty: raise RuntimeError(f'no buy {ticker} {start}')
        b=float(bp.iloc[0].adj_open)
        if final: sp=z[z.index<=end]; e=float(sp.iloc[-1].adj_close); ed=sp.index[-1]
        else:
            sp=z[(z.index>=end)&(z.index<=end+pd.Timedelta(days=7))]
            if sp.empty: sp=z[z.index<end]; e=float(sp.iloc[-1].adj_close); ed=sp.index[-1]
            else: e=float(sp.iloc[0].adj_open); ed=sp.index[0]
        return e/b,ed,'Yahoo single'
    b=float(buy.iloc[0].adj_open)
    if final:
        sell=x[x.date<=end]; e=float(sell.iloc[-1].adj_close); ed=sell.iloc[-1].date
    else:
        sell=x[(x.date>=end)&(x.date<=end+pd.Timedelta(days=7))]
        if sell.empty: sell=x[x.date<end]; e=float(sell.iloc[-1].adj_close); ed=sell.iloc[-1].date
        else: e=float(sell.iloc[0].adj_open); ed=sell.iloc[0].date
    if not np.isfinite(e/b) or e/b<=0: raise RuntimeError(f'bad growth {ticker}')
    return e/b,ed,'panel'

def spy_growth(spy,start,end,final=False):
    b=spy[(spy.index>=start)&(spy.index<=start+pd.Timedelta(days=7))]; bp=float(b.iloc[0].adj_open)
    if final: s=spy[spy.index<=end]; ep=float(s.iloc[-1].adj_close)
    else: s=spy[(spy.index>=end)&(spy.index<=end+pd.Timedelta(days=7))]; ep=float(s.iloc[0].adj_open)
    return ep/bp

def main():
    m=get_members(); symbols=set(m.symbol)
    spy=one('SPY',pd.Timestamp('1993-01-01'),ASOF); spy=spy[spy.index<=ASOF]
    if spy.empty: raise RuntimeError('SPY missing')
    actual=spy.index.max(); print('asof',actual,flush=True)
    ev=trade_dates(spy); print('events',len(ev),flush=True)
    panel=dl(symbols,pd.Timestamp('1994-01-01'),actual)
    panel.to_parquet('panel.parquet',index=False)
    picks=[]
    for i,e in enumerate(ev):
        p=rank_event(panel,m,e); picks.append(p)
        if (i+1)%10==0: print('ranked',i+1,flush=True)
    p=pd.DataFrame(picks).sort_values(['schedule','year']).reset_index(drop=True)
    p['growth']=np.nan; p['spy_growth']=np.nan; p['exit_date']=pd.NaT; p['holding_source']=''
    for s in SCHEDULES:
        ids=p.index[p.schedule.eq(s)].tolist()
        for j,idx in enumerate(ids):
            row=p.loc[idx]; final=j==len(ids)-1; end=actual if final else p.loc[ids[j+1],'trade_date']
            g,ed,src=growth(panel,row.ticker,row.symbol,row.trade_date,end,final); p.loc[idx,['growth','exit_date','holding_source']]=[g,ed,src]; p.loc[idx,'spy_growth']=spy_growth(spy,row.trade_date,end,final)
    out=[]
    for s in SCHEDULES:
        ps=p[p.schedule.eq(s)].sort_values('year')
        for h in range(1,31):
            sy=actual.year-h; q=ps[ps.year>=sy]
            sv=bv=0.0
            for r in q.itertuples(index=False): sv=(sv+CONTRIB)*float(r.growth); bv=(bv+CONTRIB)*float(r.spy_growth)
            c=CONTRIB*len(q); out.append({'schedule':s,'horizon_years':h,'start_year':sy,'start_date':q.iloc[0].trade_date,'as_of_date':actual,'contributions':len(q),'total_contributed':c,'strategy_value':sv,'spy_value':bv,'strategy_profit':sv-c,'spy_profit':bv-c,'strategy_to_spy_ratio':sv/bv,'beats_spy':sv>bv})
    r=pd.DataFrame(out).sort_values(['horizon_years','schedule']); r.to_csv(OUT/'backtests_120.csv',index=False); p.to_csv(OUT/'annual_picks.csv',index=False)
    summary=r.groupby('schedule').agg(horizons_beating_spy=('beats_spy','sum'),median_ratio=('strategy_to_spy_ratio','median'),mean_ratio=('strategy_to_spy_ratio','mean')).reset_index(); z=r[r.horizon_years.eq(30)][['schedule','total_contributed','strategy_value','spy_value','strategy_to_spy_ratio']]; summary=summary.merge(z,on='schedule'); summary.to_csv(OUT/'summary.csv',index=False)
    meta={'requested_asof':'2026-09-01','actual_asof':str(actual.date()),'schedules':list(SCHEDULES),'signal':'trailing 12m adjusted close ending prior trading day','execution':'first trading day on/after schedule at adjusted open','membership':'point-in-time fja05680 intervals','benchmark':'SPY adjusted prices','contribution':'$1000 at every rebalance including start and current-year rebalance','limitations':'Open-data backtest. Yahoo can lack delisted historical tickers; unavailable series are omitted from that event. Taxes/slippage/fees excluded.'}; (OUT/'metadata.json').write_text(json.dumps(meta,indent=2)); print(summary.to_string(index=False),flush=True)
if __name__=='__main__': main()
