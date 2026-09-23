import json
from pathlib import Path
import pandas as pd
import numpy as np

SEASON = 2026
MAX_ROSTERED_PCT = 60
LOOKBACK = 3

# nflreadpy is the maintained Python interface to nflverse.
# The GitHub Action installs it before running this script.
import nflreadpy as nfl

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

def safe_num(x):
    try:
        return float(x)
    except Exception:
        return 0.0

def first_existing(df, names, default=0):
    for n in names:
        if n in df.columns:
            return n
    df["_missing"] = default
    return "_missing"

def zclip(series, lo=0, hi=100):
    return series.clip(lo, hi)

def weighted_recent(group, col):
    """Latest week gets 60%, previous 25%, third 15%."""
    g = group.sort_values("week", ascending=False).head(LOOKBACK)
    if len(g) == 0:
        return 0
    weights = [0.60, 0.25, 0.15][:len(g)]
    vals = pd.to_numeric(g[col], errors="coerce").fillna(0).tolist()
    return sum(v*w for v,w in zip(vals, weights)) / sum(weights)

def trend(group, col):
    g = group.sort_values("week").tail(LOOKBACK)
    if len(g) < 2:
        return 0
    a = safe_num(g[col].iloc[-1])
    b = safe_num(g[col].iloc[:-1].mean())
    # percentage-point change, scaled into a useful 0-100 contribution
    return max(-25, min(25, (a-b)*2))

print("Loading nflverse weekly player stats...")
weekly = nfl.load_player_stats(SEASON).to_pandas()

print("Loading nflverse snap counts...")
snaps = nfl.load_snap_counts(SEASON).to_pandas()

# Normalize common identifiers.
weekly = weekly.copy()
snaps = snaps.copy()

for df in (weekly, snaps):
    for c in ["week", "season"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

# Restrict to completed regular-season weeks.
if "season_type" in weekly.columns:
    weekly = weekly[weekly.season_type.eq("REG")]
if "season_type" in snaps.columns:
    snaps = snaps[snaps.season_type.eq("REG")]

latest_week = int(weekly["week"].max())
weeks = sorted([w for w in weekly["week"].dropna().unique() if w <= latest_week])[-LOOKBACK:]
weekly = weekly[weekly.week.isin(weeks)].copy()
snaps = snaps[snaps.week.isin(weeks)].copy()

# Core player stat fields.
targets_col = first_existing(weekly, ["targets"])
carries_col = first_existing(weekly, ["carries", "rushing_attempts"])
air_col = first_existing(weekly, ["receiving_air_yards", "rec_air_yards"])
ppr_col = first_existing(weekly, ["fantasy_points_ppr", "fantasy_points"])

# Team denominators.
for c in [targets_col, carries_col, air_col]:
    weekly[c] = pd.to_numeric(weekly[c], errors="coerce").fillna(0)

weekly["target_share_calc"] = weekly[targets_col] / weekly.groupby(["week","team"])[targets_col].transform("sum").replace(0, np.nan) * 100
weekly["carry_share_calc"] = weekly[carries_col] / weekly.groupby(["week","team"])[carries_col].transform("sum").replace(0, np.nan) * 100
weekly["air_share_calc"] = weekly[air_col] / weekly.groupby(["week","team"])[air_col].transform("sum").replace(0, np.nan) * 100

# Snap data column discovery. nflverse/PFR naming can evolve, so use fallbacks.
snap_col = first_existing(snaps, ["offense_snaps", "offensive_snaps", "offense_snap"])
team_snap_col = first_existing(snaps, ["offense_snaps_team", "team_offense_snaps"])

# If team snap denominator is not directly supplied, calculate it from max player snaps per team/week.
if team_snap_col == "_missing":
    team_max = snaps.groupby(["week","team"])[snap_col].transform("max")
    snaps["snap_share_calc"] = pd.to_numeric(snaps[snap_col], errors="coerce").fillna(0) / team_max.replace(0,np.nan) * 100
else:
    snaps["snap_share_calc"] = pd.to_numeric(snaps[snap_col], errors="coerce").fillna(0) / pd.to_numeric(snaps[team_snap_col], errors="coerce").replace(0,np.nan) * 100

# Merge by name/team/week because source ID fields can differ across feeds.
name_w = first_existing(weekly, ["player_name", "player_display_name"])
name_s = first_existing(snaps, ["player", "player_name", "player_display_name"])
team_w = first_existing(weekly, ["team"])
team_s = first_existing(snaps, ["team"])

snap_keep = snaps[[c for c in ["week", team_s, name_s, "snap_share_calc"] if c in snaps.columns]].copy()
snap_keep = snap_keep.rename(columns={team_s:"team", name_s:"player_name"})
weekly["player_name"] = weekly[name_w].astype(str)
weekly["team"] = weekly[team_w].astype(str)
weekly = weekly.merge(snap_keep, on=["week","team","player_name"], how="left")

weekly["snap_share_calc"] = pd.to_numeric(weekly["snap_share_calc"], errors="coerce").fillna(0)
weekly["target_share_calc"] = weekly["target_share_calc"].fillna(0)
weekly["carry_share_calc"] = weekly["carry_share_calc"].fillna(0)
weekly["air_share_calc"] = weekly["air_share_calc"].fillna(0)

# Approximate route participation when available.
route_col = first_existing(weekly, ["routes_run", "route_participation"])
if route_col != "_missing":
    weekly["route_calc"] = pd.to_numeric(weekly[route_col], errors="coerce").fillna(0)
    # If route participation is already a percentage, retain it; otherwise
    # use routes / team pass attempts as a proxy.
    if weekly["route_calc"].max() <= 100:
        pass
    else:
        pa = first_existing(weekly, ["attempts", "passing_attempts", "pass_attempts"])
        weekly["route_calc"] = weekly["route_calc"] / weekly.groupby(["week","team"])[pa].transform("sum").replace(0,np.nan) * 100
else:
    weekly["route_calc"] = weekly["snap_share_calc"] * 0.90

# Red-zone looks from available weekly fields; if unavailable, leave at 0.
rz_candidates = [c for c in ["red_zone_targets", "redzone_targets", "rz_targets", "red_zone_opportunities"] if c in weekly.columns]
if rz_candidates:
    weekly["rz_calc"] = pd.to_numeric(weekly[rz_candidates[0]], errors="coerce").fillna(0)
else:
    weekly["rz_calc"] = 0

# Position normalization.
pos_col = first_existing(weekly, ["position"])
weekly["pos"] = weekly[pos_col].astype(str).str.upper()
weekly["pos"] = weekly["pos"].replace({"FB":"RB"})

# Keep fantasy-relevant skill positions.
weekly = weekly[weekly.pos.isin(["RB","WR","TE","QB"])].copy()

# Aggregate per player.
records = []
for (player, team, pos), g in weekly.groupby(["player_name","team","pos"]):
    latest = g.sort_values("week").iloc[-1]
    r = {
        "player": player,
        "team": team,
        "pos": pos,
        "week": int(latest_week),
        "snap": weighted_recent(g, "snap_share_calc"),
        "target": weighted_recent(g, "target_share_calc"),
        "carry": weighted_recent(g, "carry_share_calc"),
        "air": weighted_recent(g, "air_share_calc"),
        "route": weighted_recent(g, "route_calc"),
        "rz": weighted_recent(g, "rz_calc"),
        "ppr": weighted_recent(g, ppr_col) if ppr_col in g.columns else 0,
        "snap_trend": trend(g, "snap_share_calc"),
        "target_trend": trend(g, "target_share_calc"),
        "carry_trend": trend(g, "carry_share_calc"),
        "latest_snap": safe_num(latest["snap_share_calc"]),
        "latest_target": safe_num(latest["target_share_calc"]),
        "latest_rz": safe_num(latest["rz_calc"]),
    }
    records.append(r)

df = pd.DataFrame(records)

def score_row(r):
    p = r.pos.lower()
    if p == "wr":
        score = (
            r.snap*0.20 + r.route*0.15 + r.target*0.25 +
            r.air*0.10 + min(r.rz*5,100)*0.15 +
            max(0,min(100,50+r.snap_trend*2+r.target_trend*2))*0.10 +
            min(max(r.ppr*4,0),100)*0.05
        )
    elif p == "rb":
        score = (
            r.snap*0.20 + r.route*0.10 + r.carry*0.20 +
            r.target*0.10 + min(r.rz*5,100)*0.20 +
            max(0,min(100,50+r.snap_trend*2+r.carry_trend*2))*0.15 +
            min(max(r.ppr*4,0),100)*0.05
        )
    elif p == "te":
        score = (
            r.snap*0.15 + r.route*0.25 + r.target*0.25 +
            r.air*0.05 + min(r.rz*5,100)*0.20 +
            max(0,min(100,50+r.target_trend*2))*0.05 +
            min(max(r.ppr*4,0),100)*0.05
        )
    else:
        score = r.snap*0.20 + min(r.rz*5,100)*0.20 + max(0,min(100,50+r.snap_trend*2))*0.20 + min(max(r.ppr*4,0),100)*0.40
    return round(max(0,min(100,score)),1)

df["breakout_score"] = df.apply(score_row, axis=1)

def tier(s):
    if s >= 78: return "Priority Add"
    if s >= 68: return "Strong Add"
    if s >= 58: return "Speculative Add"
    if s >= 48: return "Deep-League Watch"
    return "Monitor"

df["tier"] = df.breakout_score.map(tier)

def reasons(r):
    out=[]
    if r.latest_snap >= 70: out.append("70%+ snaps")
    if r.latest_target >= 18: out.append("18%+ target share")
    if r.latest_rz >= 2: out.append("2+ RZ looks")
    if r.snap_trend >= 5: out.append("snap share rising")
    if r.target_trend >= 5: out.append("target share rising")
    if r.carry_trend >= 5: out.append("carry share rising")
    if not out: out.append("usage needs monitoring")
    return out[:4]

df["signals"] = df.apply(reasons, axis=1)

# We cannot reliably infer league roster percentages from nflverse alone.
# Keep all players and let the UI apply the user's rostered-percent threshold
# once an ownership feed is connected.
df = df.sort_values(["pos","breakout_score"], ascending=[True,False])

payload = {
    "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
    "season": SEASON,
    "through_week": latest_week,
    "weeks_used": weeks,
    "ownership_source": "Not connected — use the dashboard threshold or add an ownership API.",
    "players": df.to_dict(orient="records")
}

(DOCS/"data.json").write_text(json.dumps(payload, allow_nan=False, indent=2))
print(f"Wrote {len(df)} player records through Week {latest_week}.")
