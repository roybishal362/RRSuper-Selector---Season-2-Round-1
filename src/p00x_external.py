"""P0x - Public data (Cricsheet, Open Data Commons Attribution licence; cricsheet.org).
1. Parse T20 matches from 2022 on (T20Is incl. World Cups, Syed Mushtaq Ali, franchise leagues) from the downloaded zips.
2. Link Cricsheet people to the data pack's player IDs through matches that appear in BOTH datasets (IPL + T20Is):
   within the same match, innings and over, the bowler / striker / non-striker on matching deliveries must be the same
   person. A link needs >= 6 shared deliveries and >= 80% agreement.
3. Write matches NOT in the pack in the pack's clean schema (balls_clean_ext, matches_clean_ext,
   playing_eleven_clean_ext, players_ext), so P1/P2 can rebuild ball states for them unchanged.
Run with `python -I` (the zips are untrusted input; only json/zipfile parsing is done)."""
import sys, os, json, zipfile, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import ROOT, CFG, load, save, save_table, save_json, get_logger

log = get_logger("p00x_external")
RAWX = os.path.join(ROOT, "external", "cricsheet_raw")
COMPS = {"t20s_male": "T20I", "sma_male": "SMAT", "bbl_male": "BBL", "psl_male": "PSL", "cpl_male": "CPL",
         "sat_male": "SA20", "ilt_male": "ILT20", "mlc_male": "MLC", "bpl_male": "BPL", "lpl_male": "LPL",
         "ntb_male": "Blast", "ipl_male": "IPL"}
FULL = set(CFG["data"]["full_members"])
KO = {"final": "Final", "semi final": "Semi Final", "semi-final": "Semi Final", "qualifier 1": "Qualifier",
      "qualifier 2": "Qualifier", "qualifier": "Qualifier", "eliminator": "Eliminator", "quarter final": "Quarter Final",
      "challenger": "Qualifier", "knockout": "Eliminator", "play-off": "Qualifier", "playoff": "Qualifier"}
KIND = {"caught": "Caught", "caught and bowled": "Caught", "bowled": "Bowled", "lbw": "LBW", "run out": "RunOut",
        "stumped": "Stump", "hit wicket": "Hit Wicket", "retired out": "Retire Out", "obstructing the field": "Obstructing the field"}

# ------------------------------------------------------------ 1. parse (cached)
CACHE = os.path.join(ROOT, "external", "parsed")
os.makedirs(CACHE, exist_ok=True)
matches, balls = [], []
seen = set()
for zname, comp in (COMPS.items() if not os.path.exists(os.path.join(CACHE, "balls.parquet")) else []):
    zf = zipfile.ZipFile(os.path.join(RAWX, f"{zname}_json.zip"))
    for fn in zf.namelist():
        if not fn.endswith(".json") or fn in seen:
            continue
        d = json.loads(zf.read(fn))
        info = d["info"]
        date = info["dates"][0]
        if date < "2022-01-01" or info.get("gender") != "male" or info.get("overs") != 20 or info.get("balls_per_over", 6) != 6:
            continue
        if info.get("match_type") not in ("T20", "IT20"):
            continue
        seen.add(fn)
        ev = info.get("event", {})
        evname = ev.get("name", comp) if isinstance(ev, dict) else comp
        stage_raw = str(ev.get("stage", "")).lower() if isinstance(ev, dict) else ""
        out = info.get("outcome", {})
        reg = info["registry"]["people"]
        t1, t2 = info["teams"]
        lvl = comp
        if comp == "T20I":
            lvl = "T20I-full" if (t1 in FULL and t2 in FULL) else "T20I-assoc"
        winner = out.get("winner") or out.get("eliminator")
        tgt = None
        mrow = {"cs_id": fn[:-5], "comp": comp, "level": lvl, "event": evname, "season": str(info.get("season")),
                "date": date, "team1": t1, "team2": t2, "venue": info.get("venue"), "city": info.get("city"),
                "toss_winner": info.get("toss", {}).get("winner"), "toss_decision": info.get("toss", {}).get("decision"),
                "winner": winner, "result": out.get("result"), "method": out.get("method"),
                "stage": KO.get(stage_raw, "League" if not stage_raw else stage_raw.title()),
                "knock_out": int(stage_raw in KO), "is_wc": int("World Cup" in evname),
                "players": {t: [reg.get(p, p) for p in info.get("players", {}).get(t, [])] for t in info["teams"]}}
        for i, inn in enumerate(d.get("innings", []), start=1):
            if inn.get("super_over"):
                continue
            if i > 2:
                break
            if i == 2 and inn.get("target"):
                mrow["target_runs"] = inn["target"].get("runs"); mrow["target_overs"] = inn["target"].get("overs")
            k = 0
            for ov in inn["overs"]:
                for j, dl in enumerate(ov["deliveries"]):
                    k += 1
                    ex = dl.get("extras", {})
                    w = [x for x in dl.get("wickets", []) if x.get("kind") != "retired hurt"]
                    balls.append({"cs_id": fn[:-5], "innings": i, "bat_team": inn["team"], "overid": ov["over"] + 1,
                                  "seq_in_over": j, "innings_ball_num": k, "striker": reg.get(dl["batter"], dl["batter"]),
                                  "non_striker": reg.get(dl["non_striker"], dl["non_striker"]),
                                  "bowler": reg.get(dl["bowler"], dl["bowler"]), "runs": dl["runs"]["batter"],
                                  "extras": dl["runs"]["extras"], "total": dl["runs"]["total"],
                                  "non_boundary": bool(dl["runs"].get("non_boundary", False)),
                                  "wides": ex.get("wides", 0), "noballs": ex.get("noballs", 0), "byes": ex.get("byes", 0),
                                  "legbyes": ex.get("legbyes", 0), "penalty": ex.get("penalty", 0),
                                  "wicket_kind": w[0]["kind"] if w else None,
                                  "player_out": reg.get(w[0]["player_out"], w[0]["player_out"]) if w else None})
        matches.append(mrow)
if matches:
    M = pd.DataFrame(matches)
    Bx = pd.DataFrame(balls)
    M.assign(players=M.players.map(json.dumps)).to_parquet(os.path.join(CACHE, "matches.parquet"), index=False)
    Bx.to_parquet(os.path.join(CACHE, "balls.parquet"), index=False)
else:
    M = pd.read_parquet(os.path.join(CACHE, "matches.parquet"))
    M["players"] = M.players.map(json.loads)
    Bx = pd.read_parquet(os.path.join(CACHE, "balls.parquet"))
log.info(f"Parsed {len(M)} matches / {len(Bx):,} deliveries from 2022 on: {M.comp.value_counts().to_dict()}")

# ------------------------------------------------------------ 2. link to the pack through shared matches
pm = load("matches_clean")
inn = load("innings")
pb = load("balls_clean", columns=["match_id", "innings", "overid", "innings_ball_num", "striker_id", "non_striker_id", "bowler_id", "is_super_over"])
pb = pb[~pb.is_super_over].sort_values(["match_id", "innings", "innings_ball_num"])
pt = inn.pivot(index="match_id", columns="innings", values="total")
cst = Bx.groupby(["cs_id", "innings"]).total.sum().unstack()
M = M.join(cst.rename(columns={1: "tot1", 2: "tot2"}), on="cs_id")
pmx = pm[["match_id", "match_date"]].assign(date=pm.match_date.dt.strftime("%Y-%m-%d")).join(pt.rename(columns={1: "p1", 2: "p2"}), on="match_id")
link_rows = []
for r in M[M.comp.isin(["IPL", "T20I"])].itertuples():
    cand = pmx[pmx.date.isin([r.date, (pd.Timestamp(r.date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                              (pd.Timestamp(r.date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")])]
    if cand.empty or pd.isna(r.tot1):
        continue
    dist = ((cand.p1 - r.tot1).abs() + (cand.p2.fillna(0) - (r.tot2 if pd.notna(r.tot2) else 0)).abs()).dropna()
    if dist.empty:
        continue
    j = dist.idxmin()
    if dist[j] <= 12:
        link_rows.append((r.cs_id, int(cand.at[j, "match_id"]), float(dist[j])))
ML = pd.DataFrame(link_rows, columns=["cs_id", "match_id", "dist"]).sort_values("dist").drop_duplicates("match_id").drop_duplicates("cs_id")
log.info(f"Matches present in both datasets: {len(ML)} (pack has {pm.match_id.nunique()})")

pairs = collections.Counter()
pk = pb.set_index(["match_id", "innings", "overid"])
cg = Bx.set_index(["cs_id", "innings", "overid"]).sort_index()
pg = pb.groupby(["match_id", "innings", "overid"])
pb_groups = {k: g for k, g in pg}
for r in ML.itertuples():
    for i in (1, 2):
        try:
            c_inn = Bx[(Bx.cs_id == r.cs_id) & (Bx.innings == i)] if False else cg.loc[(r.cs_id, i)]
        except KeyError:
            continue
        for ov, cgrp in c_inn.groupby(level=0):
            pgrp = pb_groups.get((r.match_id, i, ov))
            if pgrp is None or len(pgrp) != len(cgrp):
                continue
            for a, b in ((pgrp.striker_id.values, cgrp.striker.values), (pgrp.non_striker_id.values, cgrp.non_striker.values),
                         (pgrp.bowler_id.values, cgrp.bowler.values)):
                for x, y in zip(a, b):
                    pairs[(y, int(x))] += 1
P = pd.DataFrame([(c, p, n) for (c, p), n in pairs.items()], columns=["cs_person", "player_id", "n"])
tot = P.groupby("cs_person").n.transform("sum")
P["share"] = P.n / tot
best = P.sort_values("n", ascending=False).drop_duplicates("cs_person")
best = best[(best.n >= 6) & (best.share >= 0.8)].sort_values("n", ascending=False).drop_duplicates("player_id")
people = pd.read_csv(os.path.join(RAWX, "people.csv"), usecols=["identifier", "name"]).set_index("identifier").name
players = load("players").set_index("player_id")
best["cs_name"] = best.cs_person.map(people)
best["pack_name"] = best.player_id.map(players.name)
save_table(best, "external_player_links")
link = dict(zip(best.cs_person, best.player_id))
log.info(f"Linked {len(link)} Cricsheet people to pack player IDs (of {players.shape[0]} pack players); "
         f"median shared deliveries {best.n.median():.0f}, median agreement {best.share.median():.3f}")
log.info("Link examples:\n" + best.head(12)[["cs_name", "pack_name", "player_id", "n", "share"]].to_string())

# ------------------------------------------------------------ 3. external matches in the pack schema
X = M[~M.cs_id.isin(ML.cs_id) & (M.comp != "IPL")].reset_index(drop=True)
log.info(f"External matches (not in the pack): {len(X)} -> {X.level.value_counts().to_dict()}")
X["match_id"] = 900000 + np.arange(len(X))
Bx = Bx[Bx.cs_id.isin(X.cs_id)].merge(X[["cs_id", "match_id"]], on="cs_id")
# ids: linked pack id, else a new id
others = sorted(set(Bx.striker) | set(Bx.non_striker) | set(Bx.bowler) | set(p for ps in X.players for t in ps.values() for p in t))
new_id = {p: link.get(p, 10_000_000 + i) for i, p in enumerate(others)}
for c in ("striker", "non_striker", "bowler", "player_out"):
    Bx[c + "_id"] = Bx[c].map(new_id)
team_names = sorted(set(X.team1) | set(X.team2))
team_id = {t: 20000 + i for i, t in enumerate(team_names)}
# bowling / batting type from the pack for linked players
pbt = load("balls_clean", columns=["bowler_id", "pace_or_spin", "bowling_type"]).groupby("bowler_id").agg(
    pace_or_spin=("pace_or_spin", lambda s: s.mode().iat[0]), bowling_type=("bowling_type", lambda s: s.mode().iat[0]))
pbb = load("balls_clean", columns=["striker_id", "batting_type"]).groupby("striker_id").batting_type.agg(lambda s: s.mode().iat[0])

b = pd.DataFrame({"match_id": Bx.match_id, "innings": Bx.innings, "overid": Bx.overid, "over_ball_num": Bx.seq_in_over + 1,
                  "innings_ball_num": Bx.innings_ball_num})
b["batting_team_id"] = Bx.bat_team.map(team_id).values
opp = X.set_index("match_id")[["team1", "team2"]]
bt = Bx.bat_team.values
b["bowling_team_id"] = np.where(Bx.match_id.map(opp.team1).values == bt, Bx.match_id.map(opp.team2).map(team_id).values,
                                Bx.match_id.map(opp.team1).map(team_id).values)
b["striker_id"], b["non_striker_id"], b["bowler_id"] = Bx.striker_id.values, Bx.non_striker_id.values, Bx.bowler_id.values
b["batting_type"] = b.striker_id.map(pbb).values
b["bowling_type"] = b.bowler_id.map(pbt.bowling_type).values
b["pace_or_spin"] = b.bowler_id.map(pbt.pace_or_spin).values
first_seen = pd.concat([Bx[["match_id", "innings", "innings_ball_num", "striker_id"]].rename(columns={"striker_id": "pid"}),
                        Bx[["match_id", "innings", "innings_ball_num", "non_striker_id"]].rename(columns={"non_striker_id": "pid"})])
order = first_seen.sort_values("innings_ball_num").drop_duplicates(["match_id", "innings", "pid"])
order["pos"] = order.groupby(["match_id", "innings"]).cumcount() + 1
b = b.merge(order[["match_id", "innings", "pid", "pos"]].rename(columns={"pid": "striker_id", "pos": "batting_order"}),
            on=["match_id", "innings", "striker_id"], how="left")
b["runs"] = Bx.runs.values
b["is_four"] = (Bx.runs.values == 4) & ~Bx.non_boundary.values
b["is_six"] = (Bx.runs.values == 6) & ~Bx.non_boundary.values
b["boundary"] = b.is_four | b.is_six
b["is_wicket"] = Bx.wicket_kind.notna().values
b["dismissal_type"] = Bx.wicket_kind.map(KIND).values
b["dismissed_player_id"] = Bx.player_out_id.values
b["is_wide"] = Bx.wides.values > 0
b["is_noball"] = Bx.noballs.values > 0
b["is_bye"] = (Bx.byes.values > 0) | (Bx.legbyes.values > 0)
b["extras_type"] = np.select([b.is_wide, b.is_noball, Bx.byes.values > 0, Bx.legbyes.values > 0, Bx.penalty.values > 0],
                             ["wide", "Overstep", "byes", "legbyes", "Penalty"], "none")
b["legal"] = ~(b.is_wide | b.is_noball)
b["extra_runs"] = Bx.extras.values
b["team_runs"] = Bx.total.values
b["bowler_runs"] = Bx.runs.values + Bx.wides.values + Bx.noballs.values
b["batter_faced"] = ~b.is_wide
b["bowler_wicket"] = b.is_wicket & b.dismissal_type.isin(["Caught", "Bowled", "LBW", "Stump", "Hit Wicket"])
b["striker_out"] = b.is_wicket & (b.dismissed_player_id == b.striker_id)
b["nonstriker_out"] = b.is_wicket & (b.dismissed_player_id == b.non_striker_id)
b["is_super_over"] = False
for c in ("stroke", "delivery_style", "over_or_around", "on_or_off"):
    b[c] = "Unknown"
b["ts"] = pd.NaT

mx = pd.DataFrame({"match_id": X.match_id})
comp_key = X.comp + " " + X.season
comp_ids = {k: 9000 + i for i, k in enumerate(sorted(comp_key.unique()))}
mx["comp_id"] = comp_key.map(comp_ids).values
mx["comp_name"] = np.where(X.is_wc == 1, "ICC Men's T20 World Cup " + X.season, X.event + " " + X.season)
mx["match_number"] = np.nan
mx["match_date"] = pd.to_datetime(X.date)
mx["stage"] = X.stage.values
mx["venue_id"] = X.venue.astype("category").cat.codes.values + 5000
mx["venue_country"] = np.nan
mx["home_team_id"] = X.team1.map(team_id).values; mx["home_team"] = X.team1.values
mx["away_team_id"] = X.team2.map(team_id).values; mx["away_team"] = X.team2.values
mx["home_team_captain_id"] = np.nan; mx["away_team_captain_id"] = np.nan
mx["toss_win_team_id"] = X.toss_winner.map(team_id).values
mx["toss_opted"] = X.toss_decision.map({"field": "Fielding", "bat": "Batting"}).values
mx["man_of_the_match"] = 0
mx["first_inn_revised_overs"] = np.nan
rev = X.target_overs.notna() & (X.target_overs < 20)
mx["second_inn_revised_overs"] = np.where(rev, X.target_overs * 6, np.nan)
mx["revised_target"] = np.where(rev | (X.method.notna()), X.target_runs, np.nan)
mx["knock_out"] = X.knock_out.values
mx["win_team_id"] = X.winner.map(team_id).values
mx["result"] = np.where(X.result.eq("no result"), "No Result", np.where(X.result.eq("tie"), "Match Tied",
                        X.winner.fillna("") + " Won" + np.where(X.method.notna(), " (" + X.method.fillna("") + ")", "")))
mx["match_format"] = np.where(X.comp == "T20I", "T20I", "T20")
mx["season"] = X.season.values
mx["year"] = mx.match_date.dt.year
mx["level"] = X.level.values
mx["is_tie"] = X.result.eq("tie").values
mx["no_result"] = mx.win_team_id.isna() & ~mx.is_tie
mx["impact_era"] = 0
mx["local_hour"] = np.nan
mx["night"] = np.nan
mx["is_wc"] = X.is_wc.values

pe = []
for r in X.itertuples():
    for t, plist in r.players.items():
        for p in plist:
            pe.append({"match_id": r.match_id, "team_id": team_id[t], "player_id": new_id.get(p, link.get(p)),
                       "batting_order": np.nan, "is_played": True, "is_captain": False, "is_wk": False, "is_impact": 0, "is_impact_off": 0})
pe = pd.DataFrame(pe).dropna(subset=["player_id"]).drop_duplicates(["match_id", "team_id", "player_id"])
inv = {v: k for k, v in new_id.items()}
px = pd.DataFrame({"player_id": list(inv.keys())})
px["name"] = [players.name.get(i) if i in players.index else people.get(inv[i], inv[i]) for i in px.player_id]
px["shared_name"] = False
px["in_pack"] = px.player_id.isin(players.index)

save(b, "balls_clean_ext"); save(mx, "matches_clean_ext"); save(pe, "playing_eleven_clean_ext"); save(px, "players_ext")
cov = pd.Series({"external_matches": len(mx), "external_deliveries": len(b),
                 "deliveries_with_pack_bowler": float(b.bowler_id.isin(players.index).mean()),
                 "deliveries_with_pack_batter": float(b.striker_id.isin(players.index).mean()),
                 "bowler_type_known": float(b.pace_or_spin.notna().mean())})
save_json({**cov.to_dict(), "levels": mx.level.value_counts().to_dict(), "world_cup_matches": int(mx.is_wc.sum()),
           "world_cup_knockouts": int(((mx.is_wc == 1) & (mx.knock_out == 1)).sum()), "linked_players": len(link)}, "p00x_summary")
log.info(f"Coverage: {cov.round(3).to_dict()}")
log.info("P0x done")
