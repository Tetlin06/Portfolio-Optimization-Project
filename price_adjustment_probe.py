from huggingface_hub import hf_hub_download
import pandas as pd, numpy as np
REPO='finsaber-team/FINSABER-V2-Data'
syms=['QLGC','ANDV','TSLA','AAPL','NVDA','GNW','RX','VLO','BRCM','ATI','FDO','MEE','COG','PENN']
windows=[('QLGC','2005-12-30','2006-12-29'),('ANDV','2006-12-29','2007-12-31'),('TSLA','2019-12-31','2020-12-31'),('AAPL','2006-12-29','2007-12-31'),('GNW','2009-02-27','2010-02-26'),('RX','2007-02-28','2008-02-29')]
fs=[]
for y in range(2005,2021):
 p=hf_hub_download(REPO,f'price_daily/year={y}/part-000.parquet',repo_type='dataset')
 x=pd.read_parquet(p,columns=['date','symbol','close','adjusted_close']); x=x[x.symbol.isin(syms)]; fs.append(x)
z=pd.concat(fs);z.date=pd.to_datetime(z.date);z=z.sort_values(['symbol','date'])
z['af']=z.adjusted_close/z.close;z['af_chg']=z.groupby('symbol').af.pct_change()+1
# close is expected to be split-adjusted. Remove suspicious jumps in adjusted/close factor.
z['div_chg']=z.af_chg.where(z.af_chg.between(.8,1.25),1).fillna(1)
z['close_chg']=z.groupby('symbol').close.pct_change()+1;z['tr_chg']=z.close_chg*z.div_chg
z['tri']=z.groupby('symbol').tr_chg.transform(lambda s:s.fillna(1).cumprod())
for s,a,b in windows:
 q=z[(z.symbol==s)&(z.date>=a)&(z.date<=b)].sort_values('date');x=q.iloc[0];y=q.iloc[-1]
 print(s,x.date.date(),y.date.date(),'close',x.close,y.close,'close_ret',y.close/x.close-1,'adj',x.adjusted_close,y.adjusted_close,'adj_ret',y.adjusted_close/x.adjusted_close-1,'af_ratio',y.af/x.af,'clean_tri_ret',y.tri/x.tri-1,'bad_jumps',int((~q.af_chg.between(.8,1.25)).sum()-1),flush=True)
 print(q.loc[~q.af_chg.between(.8,1.25),['date','close','adjusted_close','af','af_chg']].tail(10).to_string(index=False),flush=True)
