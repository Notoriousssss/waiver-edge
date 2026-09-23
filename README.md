# Waiver Edge — NFL Fantasy Opportunity Dashboard

A weekly-updating fantasy football waiver dashboard built around **opportunity**, not just fantasy points.

### What it tracks
- Snap share
- Target share
- Carry share
- Air-yard share
- Red-zone looks
- Recent usage trend
- Half-PPR production
- A position-specific **Breakout Score (0–100)**
- Waiver priority tiers
- Confidence flags explaining why a player is rising

### Data
The updater uses the open `nflverse` ecosystem. nflverse publishes automated player statistics and PFR snap-count data during the season. See the cited sources in the project notes.

### Important limitation
A truly automatic "waiver" ranking needs a live roster-percentage feed. This project therefore has a configurable `MAX_ROSTERED_PCT` setting. If you connect an ownership provider later, the ranking can automatically exclude players above that threshold.

### Weekly schedule
The GitHub Action runs every Wednesday morning Eastern Time after the previous week's games have settled, and can also be run manually.

### Deploy
1. Create a new GitHub repository.
2. Upload all files in this folder.
3. Enable GitHub Pages using the repository's `main` branch and `/docs` folder.
4. The Action updates `docs/data.json` weekly.
5. Open the Pages URL.

The site works without a server: the dashboard reads `docs/data.json`.

### Ranking philosophy
This is intentionally **not** a copied expert ranking. It is an opportunity model designed to surface players whose roles are becoming fantasy-relevant before box-score production fully catches up.

It is especially useful for a 12-team half-PPR league.
