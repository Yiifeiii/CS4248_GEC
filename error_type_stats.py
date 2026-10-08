import collections, sys, os
base=os.path.expanduser("~/mnt/CS4248/GEC/CS4248_GEC/")
files={"fce.train":"fce/m2/fce.train.gold.bea19.m2","fce.dev":"fce/m2/fce.dev.gold.bea19.m2",
"lang8":"lang8.bea19/lang8.train.auto.bea19.m2","nucle":"nucle/bea2019/nucle.train.gold.bea19.m2",
"wiA":"wi+locness/m2/A.train.gold.bea19.m2","wiB":"wi+locness/m2/B.train.gold.bea19.m2","wiC":"wi+locness/m2/C.train.gold.bea19.m2",
"wi.dev(ABCN)":"wi+locness/m2/ABCN.dev.gold.bea19.m2"}
stats={}
for k,f in files.items():
    c=collections.Counter(); ns=0; nerr=0; seen=False; cur=False
    for line in open(base+f,encoding="utf-8"):
        if line.startswith("S "):
            ns+=1; cur=False
        elif line.startswith("A "):
            p=line.rstrip("\n").split("|||")
            if p[-1].strip()!="0": continue
            t=p[1]
            if t=="noop": continue
            c[t]+=1; cur=True
        elif line.strip()=="" and cur: nerr+=1; cur=False
    stats[k]=(ns,nerr,c)
import json
json.dump({k:[v[0],v[1],dict(v[2])] for k,v in stats.items()},open(os.path.expanduser("~/errstats.json"),"w"))
for k,(ns,nerr,c) in stats.items():
    tot=sum(c.values())
    print(f"{k}: sents={ns} erroneous_sents={nerr} ({nerr/ns:.1%}) edits={tot}")
