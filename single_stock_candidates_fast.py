from __future__ import annotations
from pathlib import Path
import json, time, re
import numpy as np
import pandas as pd
import yfinance as yf
from huggingface_hub import hf_hub_download
import pitindex

OUT=Path('single_stock_candidates'); OUT.mkdir(exist_ok=True)
HF_REPO='finsaber-team/FINSABER-V2-Data'
ASOF=pd.Timestamp('2026-09-01')
SCHEDULES={'Jan 1':(1,1),'Mar 1':(3,1),'May 1':(5,1),'Sep 1':(9,1)}
START_YEAR=2006

def symnorm(x): return str(x).strip().upper().replace('.','-')
def ciknorm(x):
    if x is None or pd.isna(x): return None
    s=re.sub(r'\D','',str(x)); return s.zfill(10) if s else None

def download_hf():
    fs=[]
    for y in range(2004,2026):
        print('HF',y,flush=True)
        p=hf_hub_download(HF_REPO,f'price_daily/year={y}/part-000.parquet',repo_type='dataset')
        x=pd.read_parquet(p,columns=['date','symbol','cik','open','high','low','close','adjusted_close','volume'])
        x['date']=pd.to_datetime(x.date); x['symbol']=x.symbol.map(symnorm); x['cik']=x.cik.map(ciknorm); fs.append(x)
    return pd.concat(fs,ignore_index=True)

def yf_download(symbols,start,end):
    pieces=[]; symbols=sorted(set(symnorm(s) for s in symbols))
    for i in range(0,len(symbols),80):
        b=symbols[i:i+80]; print('YF',i,'/',len(symbols),flush=True)
        raw=yf.download(b,start=start.strftime('%Y-%m-%d'),end=(end+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),auto_adjust=False,actions=False,group_by='ticker',threads=True,progress=False,timeout=60)
        for s in b:
            try:
                x=raw[s].copy() if isinstance(raw.columns,pd.MultiIndex) and s in raw.columns.get_level_values(0) else raw.xs(s,level=1,axis=1).copy()
            except Exception: continue
            x.columns=[str(c).lower().replace(' ','_') for c in x.columns]
            if not {'open','high','low','close'}.issubset(x.columns): continue
            x['adjusted_close']=x.get('adj_close',x['close']); x['date']=pd.to_datetime(x.index).tz_localize(None).normalize(); x['symbol']=s; x['cik']=None
            pieces.append(x[['date','symbol','cik','open','high','low','close','adjusted_close','volume']].reset_index(drop=True))
        time.sleep(.2)
    return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()

def spy_data():
    x=yf.download('SPY',start='2004-01-01',end='2026-09-03',auto_adjust=False,actions=False,progress=False,threads=False,timeout=60)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    x.columns=[str(c).lower().replace(' ','_') for c in x.columns]; x.index=pd.to_datetime(x.index).tz_localize(None).normalize(); x['adjusted_close']=x.get('adj_close',x['close'])
    return x[['open','close','adjusted_close','volume']].dropna(subset=['adjusted_close'])

def event_grid(spy):
    dates=spy.index.sort_values(); out=[]
    for sched,(m,d) in SCHEDULES.items():
        for y in range(START_YEAR,ASOF.year+1):
            cal=pd.Timestamp(y,m,d); f=dates[(dates>=cal)&(dates<=ASOF)]
            if not len(f): continue
            trade=f[0]; signal=dates[dates<trade][-1]; target=signal-pd.DateOffset(years=1); lb=dates[dates<=target][-1]
            out.append({'schedule':sched,'year':y,'calendar_date':cal,'trade_date':trade,'signal_date':signal,'lookback_date':lb})
    return out

def endpoint(g,day,days=8):
    q=g[(g.date<=day)&(g.date>=day-pd.Timedelta(days=days))]
    return None if q.empty else q.sort_values('date').iloc[-1]

def patched_roster(day):
    r=pitindex.get_constituents(day.date()).copy()
    # Official Aug. 18, 2026 change announced Aug. 13: RDDT replaces AVB.
    if day>=pd.Timestamp('2026-08-18'):
        r=r[r.ticker!='AVB']
        if 'RDDT' not in set(r.ticker): r=pd.concat([r,pd.DataFrame([{'ticker':'RDDT','name':'Reddit','cik':'0001713445','gics_sector':None,'gics_sub_industry':None}])],ignore_index=True)
    r['symbol']=r.ticker.map(symnorm); r['pit_cik']=r.cik.map(ciknorm)
    return r

def main():
    spy=spy_data(); events=event_grid(spy); px=download_hf()
    current=set()
    for e in events:
        if e['year']==2026: current.update(patched_roster(e['signal_date']).symbol)
    y26=yf_download(current,pd.Timestamp('2025-01-01'),ASOF)
    if not y26.empty:
        cm=patched_roster(pd.Timestamp('2026-08-31')); cmap={r.symbol:r.pit_cik for r in cm.itertuples()}; y26['cik']=y26.symbol.map(cmap)
        px=pd.concat([px,y26[y26.date>=pd.Timestamp('2026-01-01')]],ignore_index=True)
    px=px.sort_values(['symbol','date']).drop_duplicates(['symbol','date'],keep='last')
    groups={s:g for s,g in px.groupby('symbol',sort=False)}
    rows=[]; cov=[]
    for i,e in enumerate(events,1):
        r=patched_roster(e['signal_date']); lbset=set(patched_roster(e['lookback_date']).symbol)
        scores=[]; exact_cik=0; symbol_covered=0
        for z in r.itertuples(index=False):
            g=groups.get(z.symbol)
            if g is None: continue
            a=endpoint(g,e['lookback_date']); b=endpoint(g,e['signal_date'])
            if a is None or b is None or a.adjusted_close<=0 or b.adjusted_close<=0: continue
            w=g[(g.date>e['lookback_date'])&(g.date<=e['signal_date'])]
            if len(w)<230: continue
            symbol_covered+=1; pc=ciknorm(b.cik)
            if z.pit_cik and pc==z.pit_cik: exact_cik+=1
            adv=float((w.tail(20).close*w.tail(20).volume).median())
            scores.append({'ticker':z.ticker,'symbol':z.symbol,'pit_name':z.name,'pit_cik':z.pit_cik,'price_cik':pc,'signal_return':float(b.adjusted_close/a.adjusted_close-1),'observations':len(w),'adv20':adv,'strict_member_at_lookback':z.symbol in lbset,'endpoint_start':a.date,'endpoint_end':b.date,'exact_cik_match':bool(z.pit_cik and pc==z.pit_cik)})
        q=pd.DataFrame(scores)
        if q.empty: raise RuntimeError(f'no scores {e}')
        q=q.sort_values('signal_return',ascending=False).head(20)
        for rank,z in enumerate(q.to_dict('records'),1): rows.append({**e,**z,'raw_rank':rank})
        cov.append({**e,'roster_size':len(r),'full_history_symbol_coverage':symbol_covered,'coverage_pct':symbol_covered/len(r),'exact_cik_matches_among_covered':exact_cik})
        print(i,'/',len(events),e['schedule'],e['year'],q.iloc[0].symbol,round(q.iloc[0].signal_return,4),'coverage',symbol_covered,'/',len(r),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'top20_candidates.csv',index=False)
    pd.DataFrame(cov).to_csv(OUT/'coverage.csv',index=False)
    (OUT/'metadata.json').write_text(json.dumps({'asof':str(ASOF.date()),'events':len(events),'rule':'PIT S&P member on prior close; full trailing 12m adjusted-close history; top raw exact-symbol candidates require issuer verification','pitindex':pitindex.info(),'hf_repo':HF_REPO},indent=2,default=str))

if __name__=='__main__': main()
