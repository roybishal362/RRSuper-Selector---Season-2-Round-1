"""P0 - Data audit and cleaning.
Reads the raw CSVs (never modified), applies the cleaning rules from METHODOLOGY_PLAN.md P0,
writes clean parquet tables and outputs/audit_report.md."""
import re
import numpy as np
import pandas as pd
from common import CFG, OUT, raw, save, save_json, get_logger

log = get_logger("p00_audit")
C = CFG["data"]
lines = ["# P0 Audit report", ""]


def note(s=""):
    lines.append(s)
    log.info(s)


# ---------------------------------------------------------------- matches
m = raw("matches")
comp = raw("competitions")
teams = raw("teams")
m = m.merge(comp, on="comp_id", how="left")
m["match_date"] = pd.to_datetime(m.match_date)
m["year"] = m.match_date.dt.year
full = set(C["full_members"])
both_full = m.home_team.isin(full) & m.away_team.isin(full)
m["level"] = np.select(
    [m.comp_name.str.startswith("IPL"), m.match_format.eq("T20"), both_full],
    ["IPL", "A-team", "T20I-full"], "T20I-assoc")
res = m.result.fillna("").str.lower()
m["is_tie"] = res.str.contains("tied|super over|one over eliminator")
m["no_result"] = m.win_team_id.isna()
m["impact_era"] = (m.level.eq("IPL") & (m.year >= 2023)).astype(int)
m["knock_out"] = m.knock_out.fillna(0).astype(int)

# ---------------------------------------------------------------- balls
b = raw("ball_by_ball")
n_raw = len(b)
b["extras_type"] = b.extras_type.fillna("none")
et = b.extras_type
b["is_super_over"] = b.innings > 2
b["is_wide"] = et.eq("wide")
b["is_noball"] = et.isin(C["noball_extras"])
b["is_bye"] = et.isin(C["bye_extras"])
b["legal"] = ~et.isin(C["illegal_extras"])
b["penalty_runs"] = np.where(et.eq("Penalty"), 5, 0)
b["extra_runs"] = (b.is_wide | b.is_noball | b.is_bye).astype(int) + b.extras_2_runs + b.penalty_runs
b["team_runs"] = b.runs + b.extra_runs
# bowler charged: bat runs + wides + no-ball penalty (+ their extra runs); byes / leg-byes not charged
b["bowler_runs"] = b.runs + np.where(b.is_wide | b.is_noball, 1 + b.extras_2_runs, 0)
b["batter_faced"] = ~b.is_wide                      # no-balls count as balls faced
b["is_wicket"] = b.is_wicket.astype(bool)
b["is_four"] = b.is_four.astype(bool)
b["is_six"] = b.is_six.astype(bool)
dt = b.dismissal_type.fillna("")
b["bowler_wicket"] = b.is_wicket & ~dt.isin(C["bowler_wicket_exclude"]) & dt.ne("")
b["striker_out"] = b.is_wicket & (b.dismissed_player_id == b.striker_id)
b["nonstriker_out"] = b.is_wicket & (b.dismissed_player_id == b.non_striker_id)
b["boundary"] = b.is_four | b.is_six
b["delivery_style"] = b.delivery_style.fillna("Unknown")
b["over_or_around"] = b.over_or_around.fillna("Unknown")
b["ts"] = pd.to_datetime(b.delivery_ist_time.str.replace(" UTC", ""), errors="coerce")

note(f"Raw ball rows: {n_raw:,}; matches with balls: {b.match_id.nunique()}; matches in matches.csv: {len(m)}")
note(f"Super-over rows (innings 3/4): {int(b.is_super_over.sum())} in {b[b.is_super_over].match_id.nunique()} matches -> excluded from main analysis")
no_balls = sorted(set(m.match_id) - set(b.match_id))
note(f"Matches with no ball data (abandoned): {len(no_balls)}")

# ---------------------------------------------------------------- audit checks
note("")
note("## Ball-level checks")
dup = b.duplicated(["match_id", "innings", "innings_ball_num"]).sum()
note(f"- Duplicate (match, innings, ball#) keys: {dup}")
note(f"- extras1_runs is a type code: {b.groupby('extras_type').extras1_runs.unique().to_dict()}")
note(f"- extras_2_runs distribution: {b.extras_2_runs.value_counts().to_dict()}")
for c in ["runs"]:
    note(f"- `{c}` values: {b[c].value_counts().sort_index().to_dict()}")
bs = b.bowl_speed
note(f"- bowl_speed: {100 * (bs == 0).mean():.1f}% zero, max {bs.max()}, share outside 60-165 km/h among non-zero: "
     f"{100 * ((bs > 0) & ((bs < 60) | (bs > 165))).sum() / (bs > 0).sum():.1f}% -> NOT used")
note(f"- dismissals: {dt[dt.ne('')].value_counts().to_dict()}")
ns = b.nonstriker_out.sum()
note(f"- wickets where the non-striker was dismissed: {ns} ({100 * ns / b.is_wicket.sum():.1f}%) -> keyed on dismissed_player_id")
odd = b.is_wicket & ~b.striker_out & ~b.nonstriker_out
note(f"- wickets where dismissed id is neither batter: {int(odd.sum())}")

mb = b[~b.is_super_over]
ov = mb.groupby(["match_id", "innings", "overid"]).legal.sum()
last_over = mb.groupby(["match_id", "innings"]).overid.transform("max")
complete = mb[mb.overid < last_over].groupby(["match_id", "innings", "overid"]).legal.sum()
note(f"- complete overs with exactly 6 legal balls: {100 * (complete == 6).mean():.3f}% ({int((complete != 6).sum())} exceptions)")
note(f"- legal-balls-per-over exceptions: {complete[complete != 6].value_counts().to_dict()}")

# innings totals vs official margin
inn = mb.groupby(["match_id", "innings"]).agg(total=("team_runs", "sum"), wk=("is_wicket", "sum"),
                                               legal=("legal", "sum"), bat=("batting_team_id", "first")).reset_index()
i1 = inn[inn.innings == 1].set_index("match_id")
i2 = inn[inn.innings == 2].set_index("match_id")
mm = m.set_index("match_id")
chk = []
for mid in i2.index.intersection(i1.index):
    r = str(mm.at[mid, "result"]).lower()
    k = re.search(r"by\s*(\d+)\s*(run|wicket|wkt)", r)
    if not k or mm.at[mid, "is_tie"]:
        continue
    n, typ = int(k.group(1)), k.group(2)
    winner = mm.at[mid, "win_team_id"]
    if typ == "run":
        if pd.notna(mm.at[mid, "revised_target"]) and winner == i1.at[mid, "bat"]:
            pred = mm.at[mid, "revised_target"] - 1 - i2.at[mid, "total"]
        else:
            pred = i1.at[mid, "total"] - i2.at[mid, "total"]
        chk.append((mid, "runs", n, pred))
    else:
        nr = b[(b.match_id == mid) & (b.innings == 2)]
        chk.append((mid, "wkts", n, 10 - int(nr.is_wicket.sum())))
chk = pd.DataFrame(chk, columns=["match_id", "type", "official", "rebuilt"])
chk["err"] = chk.rebuilt - chk.official
note("")
note("## Innings reconstruction vs official result")
for t in ["runs", "wkts"]:
    e = chk[chk.type == t].err
    note(f"- won-by-{t}: n={len(e)}, median error {e.median():.0f}, IQR [{e.quantile(.25):.0f}, {e.quantile(.75):.0f}], "
         f"|err|<=4: {100 * (e.abs() <= 4).mean():.1f}%, exact: {100 * (e == 0).mean():.1f}%")
big = chk[chk.err.abs() > 8].merge(m[["match_id", "result", "revised_target", "first_inn_revised_overs", "second_inn_revised_overs"]], on="match_id")
note(f"- large mismatches (>8): {len(big)}")
for _, r in big.iterrows():
    note(f"  - match {r.match_id}: {r.type} official {r.official} rebuilt {r.rebuilt}; result '{r.result}', revised target {r.revised_target}")
note("- Known cause of small errors: wides/byes that ran to the boundary are stored as 1 extra run (extras_2_runs is 0 on wides/byes).")

# ---------------------------------------------------------------- time zones
first = mb.groupby("match_id").ts.min().rename("first_ts").reset_index().merge(m[["match_id", "year", "level", "venue_country"]], on="match_id")
ipl = first[first.level == "IPL"]
note("")
note("## delivery_ist_time")
for y, g in ipl.groupby("year"):
    note(f"- IPL {y}: first-ball hour (as stored) counts {g.first_ts.dt.hour.value_counts().sort_index().to_dict()}")
first["utc"] = np.where(first.year <= 2023, first.first_ts - pd.Timedelta(hours=5.5), first.first_ts)
first["utc"] = pd.to_datetime(first.utc)
off = first.venue_country.map(C["tz_offset"]).fillna(0)
first["local_hour"] = (first.utc + pd.to_timedelta(off, unit="h")).dt.hour + (first.utc + pd.to_timedelta(off, unit="h")).dt.minute / 60
first["night"] = (first.local_hour >= 16.5).astype(int)
first.loc[first.first_ts.isna(), "night"] = np.nan
note("- Rule: 2022-23 stored in IST, 2024-26 stored in UTC; converted to venue local time, night = local start >= 16:30")
note(f"- IPL night share after fix: {ipl.merge(first[['match_id','night']], on='match_id').night.mean():.2f}")
m = m.merge(first[["match_id", "local_hour", "night"]], on="match_id", how="left")

# ---------------------------------------------------------------- players
mp = raw("competition_team_player_mapping")
names = pd.concat([b[["striker_id", "striker_name"]].set_axis(["player_id", "name"], axis=1),
                   b[["non_striker_id", "non_striker_name"]].set_axis(["player_id", "name"], axis=1),
                   b[["bowler_id", "bowler_name"]].set_axis(["player_id", "name"], axis=1),
                   mp[["player_id", "player_name"]].set_axis(["player_id", "name"], axis=1)]).drop_duplicates()
per_name = names.groupby("name").player_id.nunique()
shared = per_name[per_name > 1]
note("")
note("## Player identity")
note(f"- distinct player ids: {names.player_id.nunique()}; names mapping to >1 id: {len(shared)} -> {shared.to_dict()}")
per_id = names.groupby("player_id").name.nunique()
note(f"- ids with >1 spelling: {(per_id > 1).sum()}")
players = names.sort_values("name").groupby("player_id").name.first().reset_index()
players["shared_name"] = players.name.isin(shared.index)

pe = raw("playing_eleven")
d0 = len(pe)
pe = pe.drop_duplicates(["match_id", "team_id", "player_id"])
note(f"- playing_eleven duplicate rows removed: {d0 - len(pe)}")
for c in ["is_played", "is_captain", "is_wk"]:
    pe[c] = pe[c].astype(str).str.lower().eq("true")

# ---------------------------------------------------------------- impossible values
note("")
note("## Impossible / odd values")
note(f"- balls with runs==5: {int((b.runs == 5).sum())} (overthrows; kept)")
note(f"- wides with bat runs > 0: {int((b.is_wide & (b.runs > 0)).sum())}")
note(f"- is_four but runs!=4: {int((b.is_four & (b.runs != 4)).sum())}; is_six but runs!=6: {int((b.is_six & (b.runs != 6)).sum())}")
note(f"- toss_opted missing: {int(m.toss_opted.isna().sum())}; ties: {int(m.is_tie.sum())}; no result/abandoned: {int(m.no_result.sum())}")
note(f"- competition levels (matches): {m.level.value_counts().to_dict()}")

# ---------------------------------------------------------------- save
keep = ["match_id", "innings", "overid", "over_ball_num", "innings_ball_num", "batting_team_id", "bowling_team_id",
        "striker_id", "non_striker_id", "bowler_id", "batting_type", "bowling_type", "pace_or_spin", "batting_order",
        "runs", "is_four", "is_six", "boundary", "is_wicket", "dismissal_type", "dismissed_player_id", "extras_type",
        "is_wide", "is_noball", "is_bye", "legal", "extra_runs", "team_runs", "bowler_runs", "batter_faced",
        "bowler_wicket", "striker_out", "nonstriker_out", "is_super_over", "stroke", "delivery_style",
        "over_or_around", "on_or_off", "ts"]
save(b[keep], "balls_clean")
save(m, "matches_clean")
save(players, "players")
save(pe, "playing_eleven_clean")
save(chk, "margin_check")
save_json({"n_raw": n_raw, "complete_over_6_legal_pct": float(100 * (complete == 6).mean())}, "p00_summary")
open(f"{OUT}/audit_report.md", "w", encoding="utf-8").write("\n".join(lines))
log.info("P0 done")
