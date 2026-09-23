# Waiver Edge V2 — replacement update.py
from pathlib import Path
from datetime import datetime, timezone
import json, numpy as np, pandas as pd
import nflreadpy as nfl

SEASON=2026
DOCS=Path("docs"); DOCS.mkdir(exist_ok=True)
POSITIONS={"RB","WR","TE"}
WEIGHTS={0:.50,1:.30,2:.20}

def pdx(x): return x.to_pandas() if hasattr(x,"to_pandas") else pd.DataFrame(x)
def n(s): return pd.to_numeric(s,errors="coerce").fillna(0)
def c(df,*names,default=0):
    for x in names:
        if x in df.columns:return df[x]
    return pd.Series(default,index=df.index)
def div(a,b):
    a,b=n(a),n(b)
    return pd.Series(np.where(b>0,a/b,0),index=a.index)
def normpct(s):
    x=n(s).astype(float)
    if len(x) and x.max()>1.5:x=x/100
    return x.clip(0,1)
def jd(o):
    if isinstance(o,np.integer):return int(o)
    if isinstance(o,np.floating):return float(o)
    if isinstance(o,np.bool_):return bool(o)
    if isinstance(o,(pd.Timestamp,datetime)):return o.isoformat()
    if hasattr(o,"item"):return o.item()
    return str(o)

print("Loading weekly player stats...")
w=pdx(nfl.load_player_stats(SEASON,summary_level="week"))
rename={}
for target,opts in {
 "player_id":["player_id","gsis_id"],"player_name":["player_display_name","player_name","name"],
 "team":["recent_team","team","posteam"],"position":["position","position_group"],"week":["week"]
}.items():
    if target not in w:
        for x in opts:
            if x in w: rename[x]=target; break
w=w.rename(columns=rename)
need=["player_id","player_name","team","position","week"]
miss=[x for x in need if x not in w]
if miss: raise RuntimeError(f"Missing player-stat columns {miss}; got {list(w.columns)}")
w["week"]=pd.to_numeric(w["week"],errors="coerce")
w=w.dropna(subset=["week"]); w["week"]=w["week"].astype(int)
w["position"]=w["position"].astype(str).str.upper()
w=w[w.position.isin(POSITIONS)].copy()
if w.empty: raise RuntimeError("No RB/WR/TE weekly stats found.")
latest=int(w.week.max()); first=max(1,latest-2)
w=w[w.week.between(first,latest)].copy()

for out,names in {
 "targets":("targets",),"carries":("carries","rushing_attempts"),"receptions":("receptions",),
 "rec_yards":("receiving_yards",),"rush_yards":("rushing_yards",),
 "rec_tds":("receiving_tds",),"rush_tds":("rushing_tds",),
 "air_yards":("receiving_air_yards","air_yards")
}.items(): w[out]=n(c(w,*names))
w["half_ppr"]=w.receptions*.5+(w.rec_yards+w.rush_yards)*.1+(w.rec_tds+w.rush_tds)*6
g=w.groupby(["team","week"],dropna=False)
w["target_share"]=div(w.targets,g.targets.transform("sum"))
w["carry_share"]=div(w.carries,g.carries.transform("sum"))
w["air_share"]=div(w.air_yards,g.air_yards.transform("sum"))

print("Loading snap counts...")
w["snap_share"]=0.0
try:
 s=pdx(nfl.load_snap_counts(SEASON))
 print("SNAP COLUMNS:", list(s.columns))
 print("SNAP SAMPLE:")
 print(s.head(3).to_string())
 sr={}
 for target,opts in {"player_id":["player_id","pfr_player_id"],"player_name":["player","player_name"],"team":["team"],"week":["week"]}.items():
    if target not in s:
     for x in opts:
      if x in s: sr[x]=target; break
 s=s.rename(columns=sr)
 sp=next((x for x in ["offense_pct","off_pct","offensive_snap_pct","offense_snap_pct"] if x in s),None)
 if sp:
  s["snap_new"]=normpct(s[sp])
  keys=[]
  if "player_id" in s:
   overlap=set(w.player_id.dropna().astype(str))&set(s.player_id.dropna().astype(str))
   if overlap:keys=["player_id","week"]
  if not keys and all(x in s for x in ["player_name","team","week"]):keys=["player_name","team","week"]
  if keys:
   sm=s[keys+["snap_new"]].drop_duplicates(keys)
   w=w.merge(sm,on=keys,how="left")
   w["snap_share"]=n(w.snap_new); w=w.drop(columns=["snap_new"])
except Exception as e: print("Snap warning:",e)

print("Loading play-by-play for red-zone work...")
w["rz_looks"]=0.0
try:
 p=pdx(nfl.load_pbp(SEASON)); p["week"]=pd.to_numeric(p["week"],errors="coerce")
 p=p[p.week.between(first,latest)].copy()
 p=p[n(c(p,"yardline_100",default=999))<=20]
 parts=[]
 if "rusher_player_id" in p:
  x=p[p.rusher_player_id.notna()].groupby(["rusher_player_id","week"]).size().reset_index(name="rz_rush").rename(columns={"rusher_player_id":"player_id"}); parts.append(x)
 if "receiver_player_id" in p:
  x=p[p.receiver_player_id.notna()].groupby(["receiver_player_id","week"]).size().reset_index(name="rz_tgt").rename(columns={"receiver_player_id":"player_id"}); parts.append(x)
 if parts:
  rz=parts[0]
  for x in parts[1:]:rz=rz.merge(x,on=["player_id","week"],how="outer")
  rz["rz_new"]=n(c(rz,"rz_rush"))+n(c(rz,"rz_tgt"))
  w=w.merge(rz[["player_id","week","rz_new"]],on=["player_id","week"],how="left")
  w["rz_looks"]=n(w.rz_new); w=w.drop(columns=["rz_new"])
except Exception as e: print("RZ warning:",e)

# Public nflverse feeds used here do not provide reliable route participation.
# V2 leaves it blank/zero rather than inventing a proxy.
w["route_share"]=0.0
w["ago"]=latest-w.week; w["wt"]=w.ago.map(WEIGHTS).fillna(0)
metrics=["snap_share","route_share","target_share","carry_share","air_share","rz_looks","half_ppr"]

rows=[]
for pid,z in w.groupby("player_id",dropna=False):
 z=z.sort_values("week"); den=z.wt.sum() or 1
 r={"player_id":pid,"player":z.player_name.iloc[-1],"team":z.team.iloc[-1],"pos":z.position.iloc[-1]}
 for m in metrics:r[m]=float((z[m]*z.wt).sum()/den)
 a=z.iloc[-1]; b=z.iloc[-2] if len(z)>1 else a
 for m in ["snap_share","target_share","carry_share"]:r[m+"_trend"]=float(a[m]-b[m])
 rows.append(r)
P=pd.DataFrame(rows)

def pr(x):return x.rank(pct=True,method="average").fillna(0)
P["score"]=0.0
for pos in ["RB","WR","TE"]:
 ix=P.pos.eq(pos); q=P.loc[ix]
 if q.empty:continue
 if pos=="RB": sc=22*pr(q.snap_share)+28*pr(q.carry_share)+16*pr(q.target_share)+14*pr(q.rz_looks)+10*pr(q.half_ppr)+10*pr(q.carry_share_trend+q.target_share_trend)
 elif pos=="WR": sc=24*pr(q.snap_share)+28*pr(q.target_share)+20*pr(q.air_share)+10*pr(q.rz_looks)+8*pr(q.half_ppr)+10*pr(q.target_share_trend+q.snap_share_trend)
 else: sc=30*pr(q.snap_share)+30*pr(q.target_share)+15*pr(q.rz_looks)+10*pr(q.half_ppr)+15*pr(q.target_share_trend+q.snap_share_trend)
 P.loc[ix,"score"]=sc.values
P.score=P.score.clip(0,100).round(1)
def tier(x):
 return "Priority Add" if x>=82 else "Strong Add" if x>=72 else "Stash" if x>=60 else "Monitor"
def sig(r):
 a=[]
 if r.snap_share>=.70:a.append("high snap role")
 if r.snap_share_trend>=.10:a.append("snap share rising")
 if r.target_share>=.18:a.append("strong target share")
 if r.target_share_trend>=.05:a.append("targets rising")
 if r.pos=="RB" and r.carry_share>=.45:a.append("major backfield share")
 if r.carry_share_trend>=.10:a.append("carry share rising")
 if r.air_share>=.25:a.append("air-yard role")
 if r.rz_looks>=2:a.append("red-zone role")
 return a[:3] or ["usage needs monitoring"]
P["tier"]=P.score.map(tier); P["signals"]=P.apply(sig,axis=1)
P=P[P.score>=35].sort_values(["score","target_share","carry_share"],ascending=False)

out=[]
for rank,(_,r) in enumerate(P.iterrows(),1):
 out.append({"rank":rank,"player":str(r.player),"team":str(r.team),"pos":str(r.pos),
 "score":float(r.score),"tier":str(r.tier),"snap_pct":round(float(r.snap_share)*100,1),
 "route_pct":round(float(r.route_share)*100,1),"target_pct":round(float(r.target_share)*100,1),
 "carry_pct":round(float(r.carry_share)*100,1),"air_pct":round(float(r.air_share)*100,1),
 "rz_looks":round(float(r.rz_looks),1),"half_ppr":round(float(r.half_ppr),1),
 "signals":r.signals,"rostered_pct":None,"available_leagues":[]})

payload={"season":SEASON,"week":latest,"generated_at_utc":datetime.now(timezone.utc).isoformat(),
"model":"Usage+ V2.1","players":out,"meta":{"positions":["RB","WR","TE"],"lookback_weeks":3,
"note":"Yahoo availability not connected yet. Route participation is intentionally not fabricated. Front end: V2.1."}}
(DOCS/"data.json").write_text(json.dumps(payload,allow_nan=False,indent=2,default=jd),encoding="utf-8")
print(f"Wrote {len(out)} RB/WR/TE players through Week {latest}.")
