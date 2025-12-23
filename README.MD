# ♟️ Connect Four AI League & Tournament System

A **high-performance Connect Four AI framework** built for **experimentation, benchmarking, and research**.  
This project evaluates *dozens to hundreds* of AI agents under statistically grounded tournament conditions, with a clear separation between **strategic strength** and **computational efficiency**.

> This is not just a game — it’s an **AI evaluation system**.

---

## 🚀 Highlights

- ⚙️ Fast, headless simulation engine
- 🤖 Dozens of AI architectures with parameterized variants
- 📉 Auto-pruning league tournament (scales to hundreds of agents)
- 📊 Statistically robust scoring (Wilson confidence bounds)
- ⚖️ Explicit strength vs. speed tradeoff analysis
- 📦 Reproducible runs with deterministic seeding
- 🌐 Designed for future web visualization

---

## 🎮 Game Engine

- Full **Connect Four ruleset**
- Headless game loop for maximum throughput
- Deterministic RNG seeding for reproducibility
- Opening randomization to avoid trivial or solved openings

---

## 🤖 AI Agents

Implemented agent families:

- `Random`, `WeightedRandom`
- `Greedy`
- `TacticalGreedy`
- `Heuristic`
- `Minimax`
- `Expectiminimax`
- `Beam Search`
- `Beam Search v2` (shaping + move ordering)
- `Monte Carlo Tree Search (MCTS)`

Each agent supports **parameterized variants**:
- Search depth
- Temperature
- Time limits
- Exploration constants
- Heuristic weights

➡️ A single league run can easily include **hundreds of unique agents**.

---

## 🏆 League Tournament System  
### Auto-Pruning Tournament

Instead of an infeasible full round-robin, agents compete in **staged tournaments**:

1. Large initial agent pool  
2. Randomized pairings per stage  
3. Statistical evaluation after each stage  
4. Automatic pruning of weaker agents  
5. Final round-robin among top performers  

✅ Keeps runtime tractable  
✅ Preserves competitive accuracy  
✅ Avoids early noise dominating results  

---

## 📐 Scoring & Metrics

The system **explicitly separates strength from speed**.

---

### 💪 Strength (Primary Metric)

Computed using a **Wilson Lower Confidence Bound (LCB)** over points-per-game:

Strength = WilsonLCB(PPG, z)


**Why Wilson LCB?**
- Penalizes lucky short runs
- Rewards consistency
- Statistically conservative
- Resistant to variance

---

### ⚡ Efficiency (Strength × Speed)

To avoid rewarding only trivial fast agents:

Efficiency =
Strength × max(
speed_min_factor,
1 / (1 + speed_alpha × sqrt(ms_per_move / ms_target))
)


**Key properties**
- Fast agents are rewarded
- Slow but strong agents remain competitive
- Prevents “Greedy always wins” artifacts
- Makes compute-efficiency visible

---

### 📊 Additional Tracked Stats

- Wins / Draws / Losses
- Games played
- Average milliseconds per move
- Node expansions
- Search depth
- Pareto frontier (strength vs. speed)

---

## 🧪 CLI Usage

Run the project:

```bash
python -m connect4

Select:

Run AI League (auto-prune tournament)

You’ll be prompted for:

    Games per pairing

    Pairings per team per stage

    Minimum games before pruning

    Keep fraction

    Final survivor count

    Wilson Z value

    Speed penalty parameters

    Parallel worker count

✔️ Sensible defaults are provided for quick testing.
🧠 Recommended Final Run Settings

(Research-grade / conclusive data)
Setting	Value
Games per pairing	4
Pairings per team per stage	12
Min games before prune	48
Keep fraction	0.50
Final keep	50
Max stages	5–6
Wilson Z	1.96
ms_target	50
speed_alpha	0.35
speed_min_factor	0.75
Workers	6
Batch size	16

⏱️ Expected runtime: hours, not minutes.
📤 Output

At the end of a league run:

    Full ranking by Strength

    Full ranking by Efficiency

    Pareto frontier (speed vs. strength)

    CSV export of all agent statistics

Example:

Wrote CSV: league_results_YYYYMMDD_HHMMSS.csv

🗂️ Project Structure

connect4/
├── ai/        # Agent implementations
├── core/      # Board + rules
├── game/      # Game state
├── scripts/
│   ├── league.py        # Tournament engine
│   ├── league_main.py   # CLI entry
├── ui/        # Terminal UI
└── main.py

🎯 Design Goals

    Fair AI evaluation

    Statistical rigor

    Transparent speed vs. strength tradeoffs

    Scalable experimentation

    Web-ready architecture

🔮 Future Roadmap

Planned extensions:

    🌐 Web UI (agent selection, live matches)

    📊 Interactive charts (Pareto plots, Elo-like curves)

    🎥 Game replays

    🔌 API backend (FastAPI)

    🧠 Self-play training hooks

🤔 Why This Is Interesting

This system answers deeper questions than “who wins?”:

    Which agents scale well?

    Which strategies are compute-efficient?

    Where is the Pareto frontier of intelligence vs. speed?

➡️ It’s a benchmarking framework, not just a game.
👤 Author

Jacob Hamling
University of Iowa
Informatics · Analytics · Finance

Focus areas:

    AI systems

    Evaluation methodology

    Applied decision modeling


If you want, I can also:
- tighten this into a **GitHub-style README**
- add **badges + quick-start**
- rewrite sections to sound more **academic / research-paper-adjacent**
- prep a **project website landing page version**
