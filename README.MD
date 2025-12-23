Connect Four AI League & Tournament System

A high-performance Connect Four AI framework featuring dozens of agent architectures, an auto-pruning league tournament, statistically grounded scoring, and reproducible benchmarking. Designed for experimentation, research, and eventual web visualization.

Features
🎮 Game Engine

Full Connect Four ruleset

Headless game loop for fast simulation

Deterministic seeding for reproducibility

Opening randomization to avoid trivial openings

🤖 AI Agents

Implemented agents include:

Random / WeightedRandom

Greedy

TacticalGreedy

Heuristic

Minimax

Expectiminimax

Beam Search

Beam Search v2 (with shaping + ordering)

MCTS

Each agent supports multiple parameterized variants (depth, temperature, time limits, exploration constants, etc.), resulting in hundreds of competing agents per league run.

League Tournament System
Auto-Pruning Tournament

Instead of a single massive round-robin, agents compete in staged tournaments:

Large initial roster

Randomized pairings per stage

Statistical evaluation

Automatic pruning of weaker agents

Final round-robin among top performers

This keeps runs tractable while preserving competitive accuracy.

Scoring & Metrics

The system intentionally separates strength from speed.

Strength (Primary Skill Metric)

Uses a Wilson Lower Confidence Bound (LCB) over points-per-game:

Strength = WilsonLCB(PPG, z)


Why:

Penalizes lucky short runs

Rewards consistency

Statistically robust

Efficiency (Strength + Speed)

A soft speed penalty is applied to avoid rewarding only trivial fast agents:

Efficiency = Strength × max(
    speed_min_factor,
    1 / (1 + speed_alpha × sqrt(ms_per_move / ms_target))
)


Key properties:

Fast agents get rewarded

Slow but strong agents are not unfairly crushed

Prevents “Greedy always wins” artifacts

Additional Tracked Stats

Wins / Draws / Losses

Games played

Average milliseconds per move

Node counts

Search depth

Pareto frontier (strength vs speed)

CLI Usage

Run the project:

python -m connect4


Select:

3) Run AI League (auto-prune tournament)


You’ll be prompted for:

Games per pairing

Pairings per team per stage

Minimum games before pruning

Keep fraction

Final survivor count

Wilson Z value

Speed penalty parameters

Parallel worker count

Defaults are provided for quick testing.

Recommended Final Run Settings (Conclusive Data)

For a research-grade run:

Setting	Value
Games per pairing	4
Pairings per team per stage	12
Min games before prune	48
Keep fraction	0.50
Final keep	50
Max stages	5–6
Z (Wilson LCB)	1.96
ms_target	50
speed_alpha	0.35
speed_min_factor	0.75
Workers	6
Batch size	16

Expect runtime: hours, not minutes.

Output

At the end of a league run:

Full ranking by Strength

Full ranking by Efficiency

Pareto frontier (speed vs strength)

CSV export of all agent statistics

Example:

Wrote CSV: league_results_YYYYMMDD_HHMMSS.csv

Project Structure
connect4/
├── ai/                 # Agent implementations
├── core/               # Board + rules
├── game/               # Game state
├── scripts/
│   ├── league.py       # Tournament engine
│   ├── league_main.py  # CLI entry
├── ui/                 # Terminal UI
└── __main__.py

Design Goals

Fair AI evaluation

Statistical rigor

Speed/strength tradeoff visibility

Scalable experimentation

Web-ready architecture

Future Roadmap

Planned extensions:

🌐 Web UI (agent selection, live games)

📊 Interactive charts (Elo-like plots, Pareto graphs)

🎥 Game replays

🔌 API backend (FastAPI)

🧠 Self-play training hooks

Why This Is Interesting

This project goes beyond “who wins” and answers:

Which agents scale well?

Which strategies are compute-efficient?

Where is the Pareto frontier of intelligence vs speed?

It’s a benchmarking system, not just a game.

Author

Built by Jacob Hamling
University of Iowa — Informatics, Analytics, Finance
Focus: AI systems, evaluation, and applied decision modeling
