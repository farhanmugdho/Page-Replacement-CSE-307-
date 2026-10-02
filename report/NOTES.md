# Report notes (2–3 pages, PDF) — questions to answer in your own words

Required sections: problem framing, implementation summary, experimental setup, results, analysis of what changed.

Things worth checking against `results/` before you write the analysis (verify, don't just copy):
1. Which policy has the largest *absolute* hit-ratio drop? Is that the same policy that is *worst relative to OPT* after the shift?
   (Compare the "Drop" and "Gap to OPT" columns — a big drop can just mean it started high.)
2. Why do all non-OPT policies collapse to the same hit ratio under uniform-random access? What does that say about recency/frequency features?
3. Static vs adaptive decision tree: does retraining help after the shift, and what does it cost before the shift?
4. Why does logistic regression do so much worse than the tree? (Think about the shape of the decision boundary.)
5. Limitations: synthetic traces, one cache size, training labels depend on hindsight, per-eviction inference cost.
