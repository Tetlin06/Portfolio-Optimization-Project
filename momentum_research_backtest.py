from __future__ import annotations
import io, json, math, re, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
import yfinance as yf

OUT=Path('momentum_outputs'); OUT.mkdir(exist_ok=True)
BASE='https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp'
FILES={'deciles':'10_Portfolios_Prior_12_2_CSV.zip','size':'6_Portfolios_ME_Prior_12_2_CSV.zip','factors':'F-F_Research_Data_Factors_CSV.zip'}
SCHEDULES={'Jan 1':1,'Mar 1':3,'May 1':5,'Sep 1':9}
START_YEAR=1996; CONTRIBUTION=1000.0; BASE_COST=.005; VOL_TARGET=.15

def get_text(name):
    r=requests.get(f'{BASE}/{name}',timeout=120); r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        raw=z.read([n for n in z.namelist() if not n.endswith('/')][0])
    for enc in ('utf-8-sig','latin-1'):
        try:return raw.decode(enc)
        except UnicodeDecodeError:pass
    raise RuntimeError(f'Cannot decode {name}')

def parse_monthly(text,phrase=None):
    lines=text.replace('\r','').split('\n'); start=0
    if phrase:
        hits=[i for i,x in enumerate(lines) if phrase.lower() in x.lower()]
        if not hits: raise RuntimeError(f'Missing section {phrase}')
        start=hits[0]+1
    first=next((i for i in range(start,len(lines)) if re.match(r'^\s*\d{6}\s*,',lines[i])),None)
    if first is None: raise RuntimeError('No monthly rows')
    head=next((i for i in range(first-1,start-1,-1) if ',' in lines[i] and not re.match(r'^\s*\d{6}\s*,',lines[i])),None)
    if head is None: raise RuntimeError('No header')
    hdr=[x.strip() for x in lines[head].split(',')]; hdr[0]='date'
    rows=[]
    for line in lines[first:]:
        if not re.match(r'^\s*\d{6}\s*,',line):break
        rows.append(line.strip())
    df=pd.read_csv(io.StringIO(','.join(hdr)+'\n'+'\n'.join(rows)))
    df.columns=[str(c).strip().replace('  ',' ') for c in df.columns]
    df['date']=pd.to_datetime(df['date'].astype(str).str.strip(),format='%Y%m')+pd.offsets.MonthEnd(0)
    df=df.set_index('date').apply(pd.to_numeric,errors='coerce')/100
    return df.replace({-.9999:np.nan,-9.99:np.nan}).sort_index()

def pick_col(df,terms,fallback=-1):
    t=[x.lower().replace(' ','') for x in terms]
    for c in df.columns:
        n=str(c).lower().replace(' ','')
        if all(x in n for x in t):return c
    return df.columns[fallback]

def prices(symbols):
    x=yf.download(symbols,start='1993-01-01',end='2026-09-03',auto_adjust=False,actions=False,progress=False,threads=True,timeout=90)
    if x.empty:raise RuntimeError('No ETF prices')
    if isinstance(x.columns,pd.MultiIndex):
        key='Adj Close' if 'Adj Close' in x.columns.get_level_values(0) else 'Close'; x=x[key]
    else:
        key='Adj Close' if 'Adj Close' in x.columns else 'Close'; x=x[[key]]; x.columns=symbols[:1]
    x.index=pd.to_datetime(x.index).tz_localize(None)
    return x.sort_index().dropna(how='all')

def monthly_returns(px):
    x=px.resample('ME').last().pct_change(fill_method=None); x.index=x.index.to_period('M').to_timestamp('M'); return x

def net_drag(r,cost):return (1+r)*(1-cost)**(1/12)-1

def maxdd(r):
    w=(1+r.fillna(0)).cumprod(); d=w/w.cummax()-1; t=d.idxmin(); p=w.loc[:t].idxmax(); return float(d.min()),p,t

def metrics(name,r,spy,rf):
    x=pd.concat([r.rename('r'),spy.rename('b'),rf.rename('rf')],axis=1).dropna(); r=x.r; b=x.b; rf=x.rf
    yrs=len(r)/12; cagr=float((1+r).prod()**(1/yrs)-1); vol=float(r.std()*math.sqrt(12)); ex=r-rf
    mdd,peak,trough=maxdd(r); yr=(1+r).groupby(r.index.year).prod()-1; by=(1+b).groupby(b.index.year).prod()-1; common=yr.index.intersection(by.index)
    roll=(1+r).rolling(12).apply(np.prod,raw=True)-1; down=ex[ex<0]
    return {'strategy':name,'start':r.index.min(),'end':r.index.max(),'months':len(r),'cagr':cagr,'annualized_volatility':vol,
      'sharpe':float(ex.mean()/ex.std()*math.sqrt(12)),'sortino':float(ex.mean()*12/(down.std()*math.sqrt(12))) if len(down)>1 and down.std()>0 else np.nan,
      'max_drawdown':mdd,'drawdown_peak':peak,'drawdown_trough':trough,'calmar':cagr/abs(mdd),'beta_to_spy':float(r.cov(b)/b.var()),
      'correlation_to_spy':float(r.corr(b)),'positive_month_rate':float((r>0).mean()),'monthly_outperformance_rate':float((r>b).mean()),
      'annual_outperformance_rate':float((yr.loc[common]>by.loc[common]).mean()),'best_month':float(r.max()),'best_month_date':r.idxmax(),
      'worst_month':float(r.min()),'worst_month_date':r.idxmin(),'worst_12_month_return':float(roll.min()),'worst_12_month_end':roll.idxmin(),
      'historical_var_95_monthly':float(r.quantile(.05))}

def vol_managed(gross,rf):
    vol=gross.shift(1).rolling(12,min_periods=12).std()*math.sqrt(12); e=(VOL_TARGET/vol).clip(0,1).fillna(1)
    r=e*gross+(1-e)*rf-e*BASE_COST/12
    return r.rename('Momentum Large Vol15 Net'),e.rename('Momentum Exposure')

def one_backtest(ret,month,start_year,end):
    start=pd.Timestamp(start_year,month,1)+pd.offsets.MonthEnd(0); q=ret.loc[start:end]
    vals={c:0. for c in q.columns}; n=0
    for dt,row in q.iterrows():
        if dt.month==month:
            n+=1
            for c in vals:vals[c]+=CONTRIBUTION
        for c in vals: vals[c]*=1+float(row[c])
    out={'start_date':start,'end_date':end,'contributions':n,'total_contributed':n*CONTRIBUTION}; out.update(vals); return out

def run120(ret,end):
    rows=[]; strategies=[c for c in ret.columns if c!='SPY']
    for sched,month in SCHEDULES.items():
        for h in range(1,31):
            x=one_backtest(ret,month,end.year-h,end); row={'schedule':sched,'schedule_month':month,'horizon_start_year':end.year-h,'horizon_label_years':h,**x}
            for s in strategies:row[f'{s} / SPY']=row[s]/row['SPY']; row[f'{s} beats SPY']=row[s]>row['SPY']
            rows.append(row)
    z=pd.DataFrame(rows).sort_values(['horizon_label_years','schedule']).reset_index(drop=True)
    if len(z)!=120:raise RuntimeError(f'Expected 120 rows, got {len(z)}')
    return z

def summarize(bt,strategies):
    rows=[]
    for s in strategies:
        for sched,g in bt.groupby('schedule',sort=False):
            z=g[g.horizon_label_years==30].iloc[0]; ratios=g[f'{s} / SPY']
            rows.append({'strategy':s,'schedule':sched,'horizons_beating_spy':int(g[f'{s} beats SPY'].sum()),'out_of_horizons':len(g),
              'median_terminal_ratio_to_spy':float(ratios.median()),'minimum_terminal_ratio_to_spy':float(ratios.min()),'maximum_terminal_ratio_to_spy':float(ratios.max()),
              '30y_total_contributed':z.total_contributed,'30y_strategy_value':z[s],'30y_spy_value':z.SPY,'30y_ratio_to_spy':z[f'{s} / SPY']})
    return pd.DataFrame(rows)

def cost_sensitivity(gross,spy,rf,end):
    rows=[]
    for cost in [0,.0025,.005,.0065,.01,.015,.02,.0222,.03]:
        net=net_drag(gross,cost).rename('Momentum'); panel=pd.concat([net,spy.rename('SPY')],axis=1).dropna(); bt=run120(panel,end); m=metrics('Momentum',net,spy,rf)
        for sched,g in bt.groupby('schedule',sort=False):
            z=g[g.horizon_label_years==30].iloc[0]
            rows.append({'annual_cost_assumption':cost,'schedule':sched,'full_period_cagr':m['cagr'],'full_period_volatility':m['annualized_volatility'],
              'full_period_sharpe':m['sharpe'],'horizons_beating_spy':int(g['Momentum beats SPY'].sum()),'30y_strategy_value':z.Momentum,
              '30y_spy_value':z.SPY,'30y_ratio_to_spy':z['Momentum / SPY']})
    return pd.DataFrame(rows)

def live_check(px):
    d=px.loc[:'2026-09-01'].pct_change(fill_method=None)[['MTUM','SPY']].dropna(); rows=[]
    for c in ['MTUM','SPY']:
        r=d[c]; yrs=(r.index.max()-r.index.min()).days/365.25; w=(1+r).cumprod(); dd=w/w.cummax()-1
        rows.append({'asset':c,'start':r.index.min(),'end':r.index.max(),'total_return':float((1+r).prod()-1),'cagr':float((1+r).prod()**(1/yrs)-1),
          'annualized_volatility':float(r.std()*math.sqrt(252)),'max_drawdown':float(dd.min()),'best_day':float(r.max()),'worst_day':float(r.min())})
    rows[0]['beta_to_spy']=float(d.MTUM.cov(d.SPY)/d.SPY.var()); rows[0]['correlation_to_spy']=float(d.MTUM.corr(d.SPY)); rows[1]['beta_to_spy']=1.; rows[1]['correlation_to_spy']=1.
    cum=(1+d).cumprod(); cum.index.name='date'; return pd.DataFrame(rows),cum.reset_index()

def main():
    dec=parse_monthly(get_text(FILES['deciles']),'Value Weighted Returns -- Monthly')
    sm=parse_monthly(get_text(FILES['size']),'Value Weighted Returns -- Monthly')
    fac=parse_monthly(get_text(FILES['factors']))
    dc=pick_col(dec,['Hi','PRIOR']); bc=pick_col(sm,['BIG','Hi']); rc=pick_col(fac,['RF'])
    ff=pd.concat([dec[dc].rename('Momentum Top Decile Gross'),sm[bc].rename('Momentum Large Gross'),fac[rc].rename('RF')],axis=1)
    px=prices(['SPY','MTUM']); spy=monthly_returns(px).SPY.rename('SPY')
    end=min(ff.dropna().index.max(),spy.dropna().index.max()); base=pd.concat([ff,spy],axis=1).loc['1996-01-31':end].dropna(); end=base.index.max()
    base['Momentum Large Net 50bps']=net_drag(base['Momentum Large Gross'],BASE_COST); vm,expo=vol_managed(base['Momentum Large Gross'],base.RF); base[vm.name]=vm; base[expo.name]=expo
    cols=['Momentum Large Gross','Momentum Large Net 50bps','Momentum Large Vol15 Net','Momentum Top Decile Gross','SPY']; panel=base[cols].dropna(); end=panel.index.max()
    bt=run120(panel,end); strategies=cols[:-1]; summ=summarize(bt,strategies); risk=pd.DataFrame([metrics(s,panel[s],panel.SPY,base.RF) for s in cols])
    annual=(1+panel).groupby(panel.index.year).prod()-1; annual.index.name='year'; wealth=(1+panel).cumprod(); wealth.columns=[f'{c} Wealth' for c in wealth.columns]
    dd=wealth/wealth.cummax()-1; dd.columns=[c.replace(' Wealth',' Drawdown') for c in dd.columns]
    monthly=pd.concat([base[['RF','Momentum Exposure']],panel,wealth,dd],axis=1); monthly.index.name='date'
    costs=cost_sensitivity(base['Momentum Large Gross'],panel.SPY,base.RF,end)
    lm,lc=live_check(px)
    periods=[('1996-2005','1996-01-31','2005-12-31'),('2006-2015','2006-01-31','2015-12-31'),('2016-latest','2016-01-31',str(end.date())),('Global Financial Crisis','2007-10-31','2009-06-30'),('COVID 2020','2020-01-31','2020-12-31'),('2022 bear market','2022-01-31','2022-12-31')]
    sub=[]
    for label,s,e in periods:
        q=panel.loc[s:e]
        for st in cols:
            mdd,_,_=maxdd(q[st]); sub.append({'period':label,'strategy':st,'start':q.index.min(),'end':q.index.max(),'total_return':float((1+q[st]).prod()-1),'annualized_volatility':float(q[st].std()*math.sqrt(12)),'max_drawdown':mdd})
    meta={'research_end_date':str(end.date()),'live_etf_end_date':str(lm.end.max().date()),'annual_contribution':CONTRIBUTION,'schedules':SCHEDULES,'backtest_count':len(bt),
      'main_strategy':{'name':'Momentum Large Net 50bps','universe':'NYSE, AMEX and NASDAQ stocks in the Fama-French Big group, above the NYSE median market equity breakpoint','signal':'cumulative total return from months t-12 through t-2, skipping the most recent month','selection':'high group above the 70th percentile of NYSE prior returns','weighting':'value-weighted','rebalance':'monthly','dividends':'included','annual_cost_assumption':BASE_COST},
      'volatility_managed_variant':{'target_volatility':VOL_TARGET,'lookback':'12 completed months, lagged one month','max_equity_exposure':1.0,'cash_return':'Fama-French risk-free rate'},
      'selected_columns':{'top_decile':dc,'large_high':bc,'risk_free':rc},'benchmark':'SPY adjusted-close total returns on the same monthly cash-flow dates',
      'limitations':['This answers diversified long-only momentum versus SPY, not the original single-stock S&P winner rule.','Large-stock membership is a market-cap liquidity proxy; bid/ask and exact ADV are unavailable in aggregate CRSP portfolios.','Costs are assumptions and shown from 0 to 3 percent annually. Taxes are excluded.','120 research tests end at the latest common Ken French and SPY month; live MTUM versus SPY is separate through 2026-09-01.']}
    bt.to_csv(OUT/'backtests_120.csv',index=False); summ.to_csv(OUT/'backtest_summary.csv',index=False); risk.to_csv(OUT/'risk_metrics.csv',index=False)
    annual.reset_index().to_csv(OUT/'annual_returns.csv',index=False); monthly.reset_index().to_csv(OUT/'monthly_returns_and_levels.csv',index=False); costs.to_csv(OUT/'cost_sensitivity.csv',index=False)
    pd.DataFrame(sub).to_csv(OUT/'subperiod_analysis.csv',index=False); lm.to_csv(OUT/'live_etf_metrics.csv',index=False); lc.to_csv(OUT/'live_etf_cumulative.csv',index=False); (OUT/'metadata.json').write_text(json.dumps(meta,indent=2,default=str))
    print('RESEARCH END',end.date()); print('COLUMNS',meta['selected_columns']); print('\nRISK\n',risk.to_string(index=False)); print('\nSUMMARY\n',summ.to_string(index=False)); print('\n30Y\n',bt[bt.horizon_label_years==30].to_string(index=False)); print('\nLIVE\n',lm.to_string(index=False))
if __name__=='__main__':main()
