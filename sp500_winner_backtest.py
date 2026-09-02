import json, time
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
    r=requests.get(URL,timeout=60); r.raise_for_status()
    m=pd.read_csv(pd.io.common.BytesIO(r.content),dtype=str)
    m.ticker=m.ticker.str.upper().str.strip(); m.start_date=pd.to_datetime(m.start_date); m.end_date=pd.to_datetime(m.end_date).fillna(pd.Timestamp('2100-01-01')); m['symbol']=m.ticker.map(norm)
    return m

def normalize_one(x):
    if x is None or x.empty: return pd.DataFrame()
    if isinstance(x.columns,pd.MultiIndex):
        if len(x.columns.get_level_values(0).unique())==1: x.columns=x.columns.get_level_values(-1)
        elif len(x.columns.get_level_values(-1).unique())==1: x.columns=x.columns.get_level_values(0)
    x=x.copy(); x.columns=[str(c).lower().replace(' ','_') for c in x.columns]
    if 'adj_close' not in x: x['adj_close']=x.get('close')
    if not {'open','close','adj_close'}.issubset(x.columns): return pd.DataFrame()
    x['adj_open']=np.where(x['close'].abs()>1e-12,x['open']*x['adj_close']/x['close'],np.nan)
    x.index=pd.to_datetime(x.index).tz_localize(None).normalize()
    return x[['adj_open','adj_close']]

def one(symbol,start,end):
    for attempt in range(4):
        try:
            x=yf.download(symbol,start=start.strftime('%Y-%m-%d'),end=(end+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),auto_adjust=False,actions=False,progress=False,threads=False,timeout=45)
            x=normalize_one(x)
            if not x.empty: return x
        except Exception: pass
        time.sleep(1.5*(attempt+1))
    return pd.DataFrame()

def dl(symbols,start,end):
    symbols=sorted(set(symbols)); pieces=[]; failed=[]
    for i in range(0,len(symbols),60):
        batch=symbols[i:i+60]; print(f'download {i}/{len(symbols)}',flush=True)
        raw=pd.DataFrame()
        for attempt in range(3):
            try:
                raw=yf.download(batch,start=start.strftime('%Y-%m-%d'),end=(end+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),auto_adjust=False,actions=False,group_by='ticker',threads=True,progress=False,timeout=60)
                if not raw.empty: break
            except Exception: pass
            time.sleep(2*(attempt+1))
        for s in batch:
            x=pd.DataFrame()
            try:
                if isinstance(raw.columns,pd.MultiIndex):
                    if s in raw.columns.get_level_values(0): x=raw[s]
                    elif s in raw.columns.get_level_values(1): x=raw.xs(s,level=1,axis=1)
                elif len(batch)==1: x=raw
            except Exception: pass
            x=normalize_one(x)
            if x.empty:
                failed.append(s); continue
            z=x.reset_index(); z=z.rename(columns={z.columns[0]:'date'}); z['symbol']=s; pieces.append(z[['date','symbol','adj_open','adj_close']])
        time.sleep(.25)
    print('symbols with no batch series',len(failed),flush=True)
    if not pieces: raise RuntimeError('No Yahoo stock panel')
    return pd.concat(pieces,ignore_index=True).drop_duplicates(['date','symbol'],keep='last')

def trade_dates(spy):
    dates=spy.index.sort_values(); ev=[]
    for name,(mo,da) in SCHEDULES.items():
        for y in range(START,ASOF.year+1):
            cal=pd.Timestamp(y,mo,da); fut=dates[(dates>=cal)&(dates<=ASOF)]
            if not len(fut): continue
            td=fut[0]; pos=dates.get_loc(td)
            if pos==0: continue
            sig=dates[pos-1]; target=sig-pd.DateOffset(years=1); past=dates[dates<=target]
            if len(past): ev.append((name,y,cal,td,sig,past[-1]))
    return ev

def first_row(df,date,days=7):
    q=df[(df.index>=date)&(df.index<=date+pd.Timedelta(days=days))]
    return None if q.empty else q.iloc[0]

def last_row(df,date,days=10):
    q=df[(df.index<=date)&(df.index>=date-pd.Timedelta(days=days))]
    return None if q.empty else q.iloc[-1]

def main():
    m=get_members(); symbols=sorted(set(m.symbol))
    spy=one('SPY',pd.Timestamp('1993-01-01'),ASOF); spy=spy[spy.index<=ASOF]
    if spy.empty: raise RuntimeError('SPY missing')
    actual=spy.index.max(); print('asof',actual.date(),flush=True)
    events=trade_dates(spy); print('events',len(events),flush=True)
    panel=dl(symbols,pd.Timestamp('1994-01-01'),actual)
    close=panel.pivot(index='date',columns='symbol',values='adj_close').sort_index()
    opn=panel.pivot(index='date',columns='symbol',values='adj_open').sort_index()
    available=set(close.columns)
    print('panel',close.shape,'available symbols',len(available),flush=True)

    picks=[]; missing_by_event=[]
    for i,(name,y,cal,td,sig,lb) in enumerate(events,1):
        eligible=m[(m.start_date<=td)&(m.end_date>td)][['ticker','symbol']].drop_duplicates('ticker')
        eligible=eligible[eligible.symbol.isin(available)].copy()
        lbrow=last_row(close,lb,10); sigrow=last_row(close,sig,10); buyrow=first_row(opn,td,7)
        if lbrow is None or sigrow is None or buyrow is None: raise RuntimeError(f'market endpoint missing {name} {y}')
        syms=eligible.symbol.tolist()
        a=lbrow.reindex(syms); b=sigrow.reindex(syms); o=buyrow.reindex(syms)
        window=close.loc[(close.index>lb)&(close.index<=sig),syms]
        obs=window.notna().sum(); dr=window.pct_change(fill_method=None); mx=dr.max(); mn=dr.min()
        ret=b/a-1
        valid=(a>0)&(b>0)&(o>0)&(obs>=200)&(mx.fillna(0)<=2.0)&(mn.fillna(0)>=-.8)&np.isfinite(ret)
        scores=ret.where(valid).dropna().sort_values(ascending=False)
        if scores.empty: raise RuntimeError(f'no valid rank {name} {y}')
        wsym=scores.index[0]; erow=eligible[eligible.symbol.eq(wsym)].iloc[0]
        picks.append({'schedule':name,'year':y,'calendar_date':cal,'trade_date':td,'signal_date':sig,'lookback_date':lb,'eligible_members':len(m[(m.start_date<=td)&(m.end_date>td)].ticker.unique()),'eligible_with_price':len(eligible),'ticker':erow.ticker,'symbol':wsym,'signal_return':float(scores.iloc[0]),'observations':int(obs[wsym])})
        missing_by_event.append({'schedule':name,'year':y,'members_missing_price':len(m[(m.start_date<=td)&(m.end_date>td)].ticker.unique())-len(eligible)})
        if i%12==0: print('ranked',i,'/',len(events),flush=True)

    p=pd.DataFrame(picks).sort_values(['schedule','year']).reset_index(drop=True)
    p['growth']=np.nan; p['spy_growth']=np.nan; p['exit_date']=pd.NaT; p['exit_method']=''
    for sched in SCHEDULES:
        ids=p.index[p.schedule.eq(sched)].tolist()
        for j,idx in enumerate(ids):
            row=p.loc[idx]; final=j==len(ids)-1; end=actual if final else pd.Timestamp(p.loc[ids[j+1],'trade_date']); sym=row.symbol
            buyrow=first_row(opn[[sym]].dropna(),pd.Timestamp(row.trade_date),7)
            if buyrow is None or not (float(buyrow[sym])>0): raise RuntimeError(f'buy missing {sched} {row.year} {row.ticker}')
            bp=float(buyrow[sym])
            if final:
                sq=close[[sym]].dropna(); sq=sq[sq.index<=end]
                if sq.empty: raise RuntimeError(f'final missing {sym}')
                ep=float(sq.iloc[-1][sym]); ed=sq.index[-1]; method='final adjusted close'
            else:
                sellrow=first_row(opn[[sym]].dropna(),end,7)
                if sellrow is not None:
                    ep=float(sellrow[sym]); ed=opn[[sym]].dropna()[(opn[[sym]].dropna().index>=end)&(opn[[sym]].dropna().index<=end+pd.Timedelta(days=7))].index[0]; method='next rebalance adjusted open'
                else:
                    sq=close[[sym]].dropna(); sq=sq[sq.index<end]
                    if sq.empty: raise RuntimeError(f'exit missing {sym}')
                    ep=float(sq.iloc[-1][sym]); ed=sq.index[-1]; method='last adjusted close then cash'
            g=ep/bp
            if not np.isfinite(g) or g<=0: raise RuntimeError(f'bad growth {sym}')
            p.loc[idx,['growth','exit_date','exit_method']]=[g,ed,method]
            sb=first_row(spy[['adj_open']].dropna(),pd.Timestamp(row.trade_date),7)
            if final: ss=spy[spy.index<=end]; sg=float(ss.iloc[-1].adj_close)/float(sb.adj_open)
            else: ss=first_row(spy[['adj_open']].dropna(),end,7); sg=float(ss.adj_open)/float(sb.adj_open)
            p.loc[idx,'spy_growth']=sg

    results=[]
    for sched in SCHEDULES:
        ps=p[p.schedule.eq(sched)].sort_values('year')
        for h in range(1,31):
            sy=actual.year-h; q=ps[ps.year>=sy]
            if q.empty or int(q.iloc[0].year)!=sy: raise RuntimeError(f'missing horizon {sched} {h}')
            sv=bv=0.0
            for r in q.itertuples(index=False): sv=(sv+CONTRIB)*float(r.growth); bv=(bv+CONTRIB)*float(r.spy_growth)
            c=CONTRIB*len(q)
            results.append({'schedule':sched,'horizon_years':h,'start_year':sy,'start_date':q.iloc[0].trade_date,'as_of_date':actual,'contributions':len(q),'total_contributed':c,'strategy_value':sv,'spy_value':bv,'strategy_profit':sv-c,'spy_profit':bv-c,'strategy_to_spy_ratio':sv/bv,'beats_spy':sv>bv})
    r=pd.DataFrame(results).sort_values(['horizon_years','schedule'])
    if len(r)!=120: raise RuntimeError(f'expected 120, got {len(r)}')
    summary=r.groupby('schedule').agg(horizons_beating_spy=('beats_spy','sum'),median_ratio=('strategy_to_spy_ratio','median'),mean_ratio=('strategy_to_spy_ratio','mean')).reset_index()
    summary=summary.merge(r[r.horizon_years.eq(30)][['schedule','total_contributed','strategy_value','spy_value','strategy_to_spy_ratio']],on='schedule')
    p.to_csv(OUT/'annual_picks.csv',index=False); r.to_csv(OUT/'backtests_120.csv',index=False); summary.to_csv(OUT/'summary.csv',index=False); pd.DataFrame(missing_by_event).to_csv(OUT/'coverage_by_event.csv',index=False)
    meta={'requested_asof':'2026-09-01','actual_asof':str(actual.date()),'schedules':list(SCHEDULES),'signal':'trailing 12-month adjusted-close return ending prior trading day','execution':'first trading day on/after schedule date, adjusted open','membership_source':URL,'membership_rule':'member on trade date','new_member_rule':'eligible if member on trade date and at least 200 daily observations in trailing window','benchmark':'SPY adjusted prices on identical cash-flow dates','contribution_rule':'$1,000 at each annual rebalance, including start and current-year rebalance if reached','taxes_fees_slippage':'excluded','coverage_note':'Open-data research backtest. Yahoo historical coverage can omit delisted/renamed securities; coverage_by_event.csv quantifies missing members.'}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('\nSUMMARY\n',summary.to_string(index=False),flush=True); print('\n30Y\n',r[r.horizon_years.eq(30)].to_string(index=False),flush=True)

if __name__=='__main__': main()
