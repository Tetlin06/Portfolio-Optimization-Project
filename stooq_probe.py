import csv, io, os, sys, zipfile
from pathlib import Path
import pandas as pd

ZIP=Path('/tmp/d_us_txt.zip')
TARGETS=['qlgc','andv','rx','ep','q','gnw','vlo','brcm','ati','fdo','mee','cog','penn','nov','hrb','nvda','tsla','aapl','spy']
with zipfile.ZipFile(ZIP) as z:
    names=z.namelist()
    print('members',len(names),'bytes',ZIP.stat().st_size,flush=True)
    lower={n.lower():n for n in names}
    for t in TARGETS:
        hits=[n for n in names if n.lower().endswith('/'+t+'.us.txt') or n.lower().endswith('/'+t+'.txt')]
        print('\nTARGET',t,'hits',hits[:20],flush=True)
        for n in hits[:2]:
            raw=z.read(n).decode('utf-8-sig',errors='replace')
            df=pd.read_csv(io.StringIO(raw))
            print(n,df.head(2).to_dict('records'),df.tail(2).to_dict('records'),'range',df.iloc[0].to_dict() if len(df) else None,flush=True)
            if '<DATE>' in df:
                df['date']=pd.to_datetime(df['<DATE>'].astype(str),format='%Y%m%d',errors='coerce')
                for a,b in [('2005-12-20','2007-01-10'),('2006-12-20','2008-01-10'),('2007-02-20','2008-03-10'),('2009-02-20','2010-03-10')]:
                    q=df[(df.date>=a)&(df.date<=b)]
                    if len(q): print(a,b,q.head(1).to_dict('records'),q.tail(1).to_dict('records'),flush=True)
