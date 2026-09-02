from __future__ import annotations
import json, re, time
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from huggingface_hub import hf_hub_download
import pitindex

OUT=Path('single_stock_audit'); OUT.mkdir(exist_ok=True)
HF_REPO='finsaber-team/FINSABER-V2-Data'
ASOF=pd.Timestamp('2026-09-01')
SCHEDULES={'Jan 1':(1,1),'Mar 1':(3,1),'May 1':(5,1),'Sep 1':(9,1)}
START_YEAR=2006
UA={'User-Agent':'Tomas Etlin academic investment research contact etlintomas6@gmail.com'}

ALIASES={'BF.B':'BF-B','BRK.B':'BRK-B'}
def symnorm(s): return str(s).strip().upper().replace('.','-')
def ciknorm(c):
    if c is None or pd.isna(c): return None
    x=re.sub(r'\D','',str(c)); return x.zfill(10) if x else None

def namenorm(s):
    if s is None or pd.isna(s): return ''
    x=re.sub(r'[^A-Z0-9 ]',' ',str(s).upper())
    toks=[t for t in x.split() if t not in {'THE','INC','INCORPORATED','CORP','CORPORATION','CO','COMPANY','PLC','LTD','LIMITED','HOLDING','HOLDINGS','GROUP','LP','LLC','NV','SA','AG'}]
    return ' '.join(toks)

def name_score(a,b):
    a=namenorm(a); b=namenorm(b)
    if not a or not b: return 0.0
    if a in b or b in a: return 1.0
    seq=SequenceMatcher(None,a,b).ratio()
    A=set(a.split()); B=set(b.split()); jac=len(A&B)/max(1,len(A|B))
    return max(seq,jac)

def sec_name(cik,session,cache):
    if cik in cache: return cache[cik]
    url=f'https://data.sec.gov/submissions/CIK{cik}.json'
    for attempt in range(3):
        try:
            r=session.get(url,headers=UA,timeout=30)
            if r.status_code==200:
                cache[cik]=r.json().get('name') or ''
                time.sleep(.11)
                return cache[cik]
            if r.status_code in (403,429): time.sleep(2+attempt*3)
            else: break
        except Exception: time.sleep(1+attempt)
    cache[cik]=''; return ''

def download_prices():
    fs=[]
    for y in range(2005,2026):
        print('HF year',y,flush=True)
        p=hf_hub_download(HF_REPO,f'price_daily/year={y}/part-000.parquet',repo_type='dataset')
        x=pd.read_parquet(p,columns=['date','symbol','cik','open','high','low','close','adjusted_close','volume'])
        x['date']=pd.to_datetime(x.date); x['symbol_raw']=x.symbol.astype(str); x['symbol']=x.symbol.map(symnorm); x['cik']=x.cik.map(ciknorm)
        fs.append(x)
    z=pd.concat(fs,ignore_index=True)
    print('HF panel',z.shape,z.date.min(),z.date.max(),'symbols',z.symbol.nunique(),'ciks',z.cik.nunique(),flush=True)
    dups=z.duplicated(['date','symbol'],keep=False)
    z[dups].sort_values(['date','symbol']).to_csv(OUT/'duplicate_date_symbol_rows.csv',index=False)
    return z

def yf_panel(symbols,start,end):
    pieces=[]
    syms=sorted(set(symnorm(s) for s in symbols))
    for i in range(0,len(syms),80):
        batch=syms[i:i+80]; print('Yahoo',i,'/',len(syms),flush=True)
        raw=yf.download(batch,start=start.strftime('%Y-%m-%d'),end=(end+pd.Timedelta(days=1)).strftime('%Y-%m-%d'),auto_adjust=False,actions=False,group_by='ticker',threads=True,progress=False,timeout=60)
        if raw.empty: continue
        for s in batch:
            try:
                if isinstance(raw.columns,pd.MultiIndex):
                    x=raw[s].copy() if s in raw.columns.get_level_values(0) else raw.xs(s,level=1,axis=1).copy()
                else: x=raw.copy()
            except Exception: continue
            x.columns=[str(c).lower().replace(' ','_') for c in x.columns]
            if not {'open','high','low','close'}.issubset(x.columns): continue
            x['adjusted_close']=x.get('adj_close',x['close'])
            x['date']=pd.to_datetime(x.index).tz_localize(None).normalize(); x['symbol']=s; x['symbol_raw']=s; x['cik']=None
            pieces.append(x[['date','symbol','cik','open','high','low','close','adjusted_close','volume','symbol_raw']].reset_index(drop=True))
        time.sleep(.25)
    return pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame()

def get_spy():
    x=yf.download('SPY',start='2004-01-01',end='2026-09-03',auto_adjust=False,actions=False,progress=False,threads=False,timeout=60)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    x.columns=[str(c).lower().replace(' ','_') for c in x.columns]
    x.index=pd.to_datetime(x.index).tz_localize(None).normalize()
    x['adjusted_close']=x.get('adj_close',x['close'])
    return x[['open','close','adjusted_close','volume']].dropna(subset=['adjusted_close'])

def events(spy):
    dates=spy.index.sort_values(); out=[]
    for sched,(mo,day) in SCHEDULES.items():
        for y in range(START_YEAR,ASOF.year+1):
            cal=pd.Timestamp(y,mo,day); fut=dates[(dates>=cal)&(dates<=ASOF)]
            if not len(fut): continue
            trade=fut[0]; pos=dates.get_loc(trade); signal=dates[pos-1]
            target=signal-pd.DateOffset(years=1); lb=dates[dates<=target][-1]
            out.append({'schedule':sched,'year':y,'calendar_date':cal,'trade_date':trade,'signal_date':signal,'lookback_date':lb})
    return out

def row_at_or_before(g,dt,days=8):
    q=g[(g.date<=dt)&(g.date>=dt-pd.Timedelta(days=days))]
    return None if q.empty else q.sort_values('date').iloc[-1]

def main():
    prices=download_prices(); spy=get_spy(); evs=events(spy)
    # Add 2026 Yahoo rows for union of rosters needed in 2026.
    roster_union=set()
    for e in evs:
        if e['year']==2026:
            roster_union.update(pitindex.get_constituents(e['signal_date'].date()).ticker.map(symnorm))
    y26=yf_panel(roster_union,pd.Timestamp('2025-01-01'),ASOF)
    if not y26.empty:
        # Map current PIT CIK where exact current ticker match exists.
        current=pitindex.get_constituents('2026-08-17'); cmap={symnorm(r.ticker):ciknorm(r.cik) for r in current.itertuples()}
        y26['cik']=y26.symbol.map(cmap)
        prices=prices[prices.date<pd.Timestamp('2026-01-01')]
        prices=pd.concat([prices,y26[y26.date>=pd.Timestamp('2026-01-01')]],ignore_index=True)
    prices=prices.sort_values(['symbol','date']).drop_duplicates(['symbol','date'],keep='last')
    groups={s:g.copy() for s,g in prices.groupby('symbol',sort=False)}

    raw_candidates=[]
    for ei,e in enumerate(evs,1):
        roster=pitindex.get_constituents(e['signal_date'].date()).copy(); roster['symbol']=roster.ticker.map(symnorm); roster['pit_cik']=roster.cik.map(ciknorm)
        lb_roster=set(pitindex.get_constituents(e['lookback_date'].date()).ticker.map(symnorm))
        scores=[]
        for r in roster.itertuples(index=False):
            g=groups.get(r.symbol)
            if g is None: continue
            a=row_at_or_before(g,e['lookback_date']); b=row_at_or_before(g,e['signal_date'])
            if a is None or b is None or a.adjusted_close<=0 or b.adjusted_close<=0: continue
            w=g[(g.date>e['lookback_date'])&(g.date<=e['signal_date'])]
            if len(w)<230: continue
            ret=float(b.adjusted_close/a.adjusted_close-1)
            adv=float((w.tail(20).close*w.tail(20).volume).median()) if len(w)>=20 else np.nan
            scores.append({'ticker':r.ticker,'symbol':r.symbol,'pit_name':r.name,'pit_cik':r.pit_cik,'price_cik':ciknorm(b.cik),'signal_return':ret,'observations':len(w),'adv20':adv,'strict_member_at_lookback':r.symbol in lb_roster,'endpoint_start':a.date,'endpoint_end':b.date})
        q=pd.DataFrame(scores).sort_values('signal_return',ascending=False).head(25)
        for rank,row in enumerate(q.itertuples(index=False),1):
            d=e.copy(); d.update(row._asdict()); d['raw_rank']=rank; raw_candidates.append(d)
        print('ranked event',ei,'/',len(evs),e['schedule'],e['year'],'raw top',q.iloc[0].symbol if len(q) else None,flush=True)
    cand=pd.DataFrame(raw_candidates)

    session=requests.Session(); cache={}
    for cik in sorted(c for c in cand.price_cik.dropna().unique() if c): sec_name(cik,session,cache)
    (OUT/'sec_names.json').write_text(json.dumps(cache,indent=2))
    cand['price_company_name']=cand.price_cik.map(cache)
    cand['name_match_score']=[name_score(a,b) for a,b in zip(cand.pit_name,cand.price_company_name)]
    cand['identity_status']=np.where(cand.pit_cik.notna() & cand.price_cik.eq(cand.pit_cik),'exact_cik',
        np.where(cand.pit_name.notna() & cand.price_company_name.ne('') & (cand.name_match_score>=.62),'name_match',
        np.where(cand.pit_name.notna() & cand.price_company_name.ne(''),'name_mismatch','unverified_missing_name')))
    cand['identity_verified']=cand.identity_status.isin(['exact_cik','name_match'])
    cand.to_csv(OUT/'candidate_audit_top25.csv',index=False)

    picks=[]
    for (sched,yr),g in cand.groupby(['schedule','year'],sort=False):
        v=g[g.identity_verified].sort_values('raw_rank')
        if v.empty:
            picks.append({'schedule':sched,'year':yr,'status':'NO_VERIFIED_PICK'})
        else:
            r=v.iloc[0].to_dict(); r['status']='verified'; r['rejected_higher_candidates']=int(r['raw_rank']-1); picks.append(r)
    p=pd.DataFrame(picks).sort_values(['schedule','year']); p.to_csv(OUT/'provisional_verified_picks.csv',index=False)
    print('\nPICKS\n',p[['schedule','year','symbol','pit_name','price_company_name','signal_return','raw_rank','identity_status','rejected_higher_candidates']].to_string(index=False),flush=True)
    print('\nSTATUS COUNTS\n',cand.identity_status.value_counts().to_string(),flush=True)
    print('\nUNVERIFIED RAW TOPS\n',cand[(cand.raw_rank<=5)&(~cand.identity_verified)][['schedule','year','raw_rank','symbol','pit_name','price_cik','price_company_name','signal_return','identity_status']].to_string(index=False),flush=True)
    meta={'events':len(evs),'start_year':START_YEAR,'asof':str(ASOF.date()),'hf_repo':HF_REPO,'pitindex_info':pitindex.info(),'note':'Provisional identity audit only; no portfolio conclusions yet.'}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2,default=str))

if __name__=='__main__': main()
