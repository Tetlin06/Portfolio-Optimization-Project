from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
import pitindex

REPO='finsaber-team/FINSABER-V2-Data'
years=[2005,2006,2008,2010,2015,2020,2024,2025]
frames=[]
for y in years:
    path=hf_hub_download(REPO, f'price_daily/year={y}/part-000.parquet', repo_type='dataset')
    pf=pq.ParquetFile(path)
    print('YEAR',y,'schema',pf.schema_arrow,'rowgroups',pf.num_row_groups)
    x=pd.read_parquet(path)
    x['date']=pd.to_datetime(x['date'])
    print(' shape',x.shape,'dates',x.date.min(),x.date.max(),'symbols',x.symbol.nunique(),'ciks',x.cik.nunique(dropna=True))
    print(' nulls',x.isna().sum().to_dict())
    print(' duplicate date/symbol',x.duplicated(['date','symbol']).sum(),'duplicate date/cik/symbol',x.duplicated(['date','cik','symbol']).sum())
    print(' sample',x.head(2).to_dict('records'))
    frames.append(x)

allx=pd.concat(frames,ignore_index=True)
for sym in ['EP','LEH','LEHMQ','BSC','BSC1','META','FB','GOOG','GOOGL','BRK.B','BRK-B','SNDK']:
    q=allx[allx.symbol.astype(str).str.upper().eq(sym)]
    if len(q): print('SYMBOL',sym,'rows',len(q),'ciks',q.cik.dropna().astype(str).unique()[:20],'range',q.date.min(),q.date.max())

for day in ['2005-01-03','2006-01-03','2008-09-12','2015-01-02','2024-01-02','2025-01-02','2026-08-17']:
    roster=pitindex.get_constituents(day)
    print('ROSTER',day,len(roster),'cik_nonnull',roster.cik.notna().sum())
    print(roster.head(3).to_dict('records'))

print('pit info',pitindex.info())
