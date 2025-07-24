# pass@k metric

**Pass\@1 is just “single‑shot functional‑correctness accuracy.”**
Here’s how the metric is defined in LiveCodeBench (and most modern code‑generation benchmarks):

| step | what happens                                                                                                                                                                                            |
| ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.   | For every programming problem in the benchmark you **sample *n* candidate solutions** from the model (LiveCodeBench uses *n*=10 by default, temperature 0.2).                                           |
| 2.   | Each candidate is executed against the hidden unit‑test suite. A candidate is **“correct”** if it passes *all* tests.                                                                                   |
| 3.   | Let ***c*** be the number of correct candidates for that problem. Pass@*k* (Kulal et al., 2020) is an *unbiased* estimate of the probability that **at least one of *k* independent samples succeeds**: |

$$
\text{pass@}k \;=\;1-\frac{\binom{n-c}{k}}{\binom{n}{k}}
$$

(implemented in `estimate_pass_at_k` of the `code_eval` metric) ([Hugging Face][1]) |

**When $k=1$ the formula collapses to $c/n$.**
So **Pass\@1 is simply the fraction of generated samples that are correct, averaged over all problems**. If you only draw one sample per problem ($n=1$), Pass\@1 reduces to the familiar “percent of problems solved on the first try.”

### Why benchmarks report Pass\@1

* **Single‑try realism:** Many users ask an LLM for code once and run it as‑is; Pass\@1 measures how often that works.
* **Baseline for sampling gains:** Comparing Pass\@1 to Pass\@5 or Pass\@10 shows how much extra mileage you get by sampling multiple times and picking the best run.
* **Leaderboard ordering:** LiveCodeBench v6 leaderboard columns (Easy/Medium/Hard) are sorted by Pass\@1 to emphasize out‑of‑the‑box reliability. ([GitHub][2])

### Practical interpretation

* **80 % Pass\@1** ⇒ out of 100 unseen contest problems, a single response from the model solves \~80.
* **80 % → 93 % gap between Pass\@1 and Pass\@5** ⇒ sampling five times and choosing any solution that passes tests can rescue another 13 problems.

In short, **“Pass\@1” answers the question, “If I let the model write code once, how often will it already be correct?”** ([medium.com][3])

[1]: https://huggingface.co/spaces/evaluate-metric/code_eval/blob/main/code_eval.py "code_eval.py · evaluate-metric/code_eval at main"
[2]: https://github.com/LiveCodeBench/LiveCodeBench "GitHub - LiveCodeBench/LiveCodeBench: Official repository for the paper \"LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code\""
[3]: https://medium.com/%40yananchen1116/a-dive-into-how-pass-k-is-calculated-for-evaluation-of-llms-coding-e52b8528235b "A dive into how pass@k is calculated for evaluation of LLM’s coding | by Yanan Chen | Medium"
