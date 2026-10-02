"""Run all policies on shifting workloads and write tables/charts to results/."""
import argparse, pathlib, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lpr.policies import FIFO, LRU, Optimal, simulate, phase_stats
from lpr.learned import LearnedPolicy, train_model
from lpr.workloads import make_trace, make_training_trace, SCENARIOS

ORDER = ["FIFO", "LRU", "OPT", "Learned-DT (static)", "Learned-LR (static)", "Learned-DT (adaptive)"]


def build_policies(frames, seed, n_train):
    train = make_training_trace(n_train, seed + 1000)  # pre-shift-style data only
    dt = train_model("dt", train, frames, seed)
    lr = train_model("lr", train, frames, seed)
    return [FIFO(), LRU(), Optimal(),
            LearnedPolicy("dt", dt, seed=seed),
            LearnedPolicy("lr", lr, seed=seed),
            LearnedPolicy("dt", dt, retrain_every=1000, window=3000, seed=seed)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--length", type=int, default=20000)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--train-length", type=int, default=10000)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(exist_ok=True)

    rows, rolling = [], {}
    t0 = time.time()
    for scen in SCENARIOS:
        for seed in range(a.seeds):
            trace, shift = make_trace(scen, a.length, seed)
            for pol in build_policies(a.frames, seed, a.train_length):
                hits = simulate(trace, a.frames, pol)
                rows.append({"scenario": scen, "seed": seed, "policy": pol.name, **phase_stats(hits, shift)})
                if seed == 0:
                    rolling[(scen, pol.name)] = hits
            print(f"[{time.time()-t0:5.0f}s] {scen} seed {seed} done", flush=True)

    df = pd.DataFrame(rows)
    # gap to Belady's optimal (per scenario/seed), and drop in hit ratio across the shift
    opt = df[df.policy == "OPT"].set_index(["scenario", "seed"])
    df["gap_to_opt_pre"] = df.apply(lambda r: opt.loc[(r.scenario, r.seed), "hit_pre"] - r.hit_pre, axis=1)
    df["gap_to_opt_post"] = df.apply(lambda r: opt.loc[(r.scenario, r.seed), "hit_post"] - r.hit_post, axis=1)
    df["hit_drop"] = df.hit_pre - df.hit_post
    df.to_csv(out / "raw_results.csv", index=False)

    cols = ["hit_pre", "hit_post", "hit_drop", "faults_pre", "faults_post", "gap_to_opt_pre", "gap_to_opt_post"]
    summ = df.groupby(["scenario", "policy"])[cols].agg(["mean", "std"])
    summ = summ.reindex(pd.MultiIndex.from_product([list(SCENARIOS), ORDER]))
    summ.to_csv(out / "summary.csv")

    # markdown table
    md = [f"Mean over {a.seeds} seeds (std in parentheses). Trace length {a.length}, shift at {a.length//2}, {a.frames} frames.\n"]
    for scen in SCENARIOS:
        md += [f"\n### Scenario: locality/sequential -> {scen}\n",
               "| Policy | Hit ratio before | Hit ratio after | Drop | Faults before | Faults after | Gap to OPT before | Gap to OPT after |",
               "|---|---|---|---|---|---|---|---|"]
        for p in ORDER:
            r = summ.loc[(scen, p)]
            f = lambda c, d=3: f"{r[(c,'mean')]:.{d}f} ({r[(c,'std')]:.{d}f})"
            md.append(f"| {p} | {f('hit_pre')} | {f('hit_post')} | {f('hit_drop')} | {f('faults_pre',0)} | {f('faults_post',0)} | {f('gap_to_opt_pre')} | {f('gap_to_opt_post')} |")
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # chart 1: hit ratio before vs after
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(13, 4.5), sharey=True)
    x = np.arange(len(ORDER)); w = 0.38
    for ax, scen in zip(axes, SCENARIOS):
        pre = [summ.loc[(scen, p)][("hit_pre", "mean")] for p in ORDER]
        post = [summ.loc[(scen, p)][("hit_post", "mean")] for p in ORDER]
        pe = [summ.loc[(scen, p)][("hit_pre", "std")] for p in ORDER]
        qe = [summ.loc[(scen, p)][("hit_post", "std")] for p in ORDER]
        ax.bar(x - w/2, pre, w, yerr=pe, label="before shift", capsize=2)
        ax.bar(x + w/2, post, w, yerr=qe, label="after shift", capsize=2)
        ax.set_xticks(x); ax.set_xticklabels([p.replace(" (", "\n(") for p in ORDER], fontsize=8)
        ax.set_title(f"locality/sequential -> {scen}"); ax.grid(axis="y", alpha=.3)
    axes[0].set_ylabel("hit ratio"); axes[0].legend()
    fig.tight_layout(); fig.savefig(out / "hit_ratio_before_after.png", dpi=150); plt.close(fig)

    # chart 2: rolling hit ratio over time (seed 0)
    fig, axes = plt.subplots(len(SCENARIOS), 1, figsize=(11, 7), sharex=True)
    k = 500
    for ax, scen in zip(axes, SCENARIOS):
        for p in ORDER:
            h = rolling[(scen, p)].astype(float)
            ax.plot(np.convolve(h, np.ones(k) / k, mode="valid"), label=p, lw=1.2)
        ax.axvline(a.length // 2 - k, color="k", ls="--", lw=1)
        ax.set_title(f"rolling hit ratio (window {k}), seed 0: locality/sequential -> {scen}")
        ax.set_ylabel("hit ratio"); ax.grid(alpha=.3)
    axes[0].legend(ncol=3, fontsize=8); axes[-1].set_xlabel("reference index (dashed line ~ shift)")
    fig.tight_layout(); fig.savefig(out / "rolling_hit_ratio.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
