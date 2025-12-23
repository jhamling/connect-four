from __future__ import annotations

import time

from .league_core import league_auto_prune
from .league_format import A
from .league_roster import build_roster

def main() -> None:

    roster = build_roster()
    print(A.bold(f"Roster size: {len(roster)} teams"))

    gpp = input("Games per pairing (default 1): ").strip()
    games_per_pair = int(gpp) if gpp else 1

    ppt = input("Pairings per team per stage (default 4): ").strip()
    stage_pairings_per_team = int(ppt) if ppt else 4

    mg = input("Min games/team before prune (default 8): ").strip()
    min_games_before_prune = int(mg) if mg else 8

    kf = input("Keep fraction each prune (default 0.60): ").strip()
    keep_fraction = float(kf) if kf else 0.60

    fk = input("Final keep (default 3): ").strip()
    final_keep = int(fk) if fk else 3

    sd = input("Seed (default 1234): ").strip()
    seed = int(sd) if sd else 1234

    mw = input("Max workers (default = cpu cores, capped at 6): ").strip()
    max_workers = int(mw) if mw else None

    bp = input("Batch pairings per worker task (default 12): ").strip()
    batch_pairings = int(bp) if bp else 12

    ms = input("Max stages (blank = no cap, recommended 3 for testing): ").strip()
    max_stages = int(ms) if ms else None

    z = input("Z for Wilson LCB (default 1.28): ").strip()
    prune_z = float(z) if z else 1.28

    mt = input("ms_target (default 50): ").strip()
    ms_target = float(mt) if mt else 50.0

    sa = input("speed_alpha (default 0.35): ").strip()
    speed_alpha = float(sa) if sa else 0.35

    sm = input("speed_min_factor (default 0.75): ").strip()
    speed_min_factor = float(sm) if sm else 0.75

    start = time.perf_counter()

    league_auto_prune(
        roster,
        games_per_pair=games_per_pair,
        stage_pairings_per_team=stage_pairings_per_team,
        min_games_before_prune=min_games_before_prune,
        keep_fraction=keep_fraction,
        final_keep=final_keep,
        seed=seed,
        max_workers=max_workers,
        batch_pairings=batch_pairings,
        max_stages=max_stages,
        prune_z=prune_z,
        ms_target=ms_target,
        speed_alpha=speed_alpha,
        speed_min_factor=speed_min_factor,
        export_csv=True,
    )

    end = time.perf_counter()
    elapsed = end - start

    h = int(elapsed // 3600)
    m = int((elapsed % 3600) // 60)
    s = elapsed % 60  

    print(A.bold(f"Total runtime: {h}:{m:02d}:{s:06.3f}"))



if __name__ == "__main__":
    main()
