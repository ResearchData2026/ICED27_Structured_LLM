# Evaluation report (ICED paper scope)

**Judge model:** gpt-5.6-luna   **Embedding model:** text-embedding-3-large  
**Generation model:** gemini-3.8-flash  
**Evaluated:** 2026-10-28 12:44  
**Concepts:** 600   **Sessions:** 150   **Briefs:** 6   **Baseline:** generic

## Key insights

- **Novelty** (chance that a method's concept beats a baseline concept): more novel than the baseline: analogy (83%), triz (73%), morphological (72%); less novel: scamper (33%).
- **Usefulness** (chance that a method's concept beats a baseline concept): less useful: scamper (37%), morphological (27%), triz (21%), analogy (16%).
- **Trade-off.** Across all concepts, novelty and usefulness correlate at rho = -0.60 (more novel concepts tend to be judged less useful).
- **Spread.** Pooled over runs, the concepts were more diverse than the baseline's in every brief for scamper (6/6 briefs).
- **Judge reliability.** Split-half reliability novelty 0.83, usefulness 0.83; reversed-order consistency rho = 0.82.

## R1. Better? Novelty and usefulness

Scores come from rankings of groups of 5 concepts within each brief (Plackett-Luce), standardised within each brief, so 0 is the average concept for that brief and 1 is one standard deviation above it. **P(beats generic)** is the probability that a concept from the method is judged better than a concept from the generic baseline for the same brief: 50% means no difference. Brackets are 95% intervals from resampling sessions.

| condition | novelty (z) | novelty: P(beats generic) | usefulness (z) | usefulness: P(beats generic) |
|---|:---:|:---:|:---:|:---:|
| generic | -0.39 | (baseline) | +0.67 | (baseline) |
| scamper | -0.81 | 33% [29, 38] | +0.29 | 37% [32, 43] |
| triz | +0.23 | 73% [67, 78] | -0.31 | 21% [16, 26] |
| analogy | +0.66 | 83% [78, 87] | -0.53 | 16% [12, 20] |
| morphological | +0.31 | 72% [68, 75] | -0.12 | 27% [23, 32] |

**Regression check** (session means, brief as a fixed effect, Holm-adjusted p for each method vs generic):

- novelty: analogy +1.05 (p = 0.000), morphological +0.70 (p = 0.000), scamper -0.42 (p = 0.002), triz +0.62 (p = 0.000)
- usefulness: analogy -1.20 (p = 0.000), morphological -0.80 (p = 0.000), scamper -0.39 (p = 0.001), triz -0.98 (p = 0.000)

**Novelty-usefulness trade-off:** Spearman rho between a concept's novelty and usefulness = -0.60 (by condition: generic -0.26, scamper -0.16, triz -0.48, analogy -0.64, morphological -0.79). Negative means more novel concepts tend to be judged less useful.

## R2. Wider? Spread of ideas

Each value is the mean over briefs; in brackets, the number of briefs where the method is higher than the generic baseline.

- **diversity_pooled**: mean semantic distance between all 20 concepts a method produced for a brief over all runs (embeddings; no judge). This is the key measure of whether a method stops the model returning to the same ideas.
- **diversity_within_session**: the same, within one session of 4 concepts. If pooled diversity is much lower than this, the method gives varied ideas in one session but repeats them across runs.

| condition | diversity_pooled | diversity_within_session |
|---|:---:|:---:|
| generic | 0.356 | 0.389 |
| scamper | 0.408 (6/6) | 0.450 (6/6) |
| triz | 0.391 (5/6) | 0.420 (5/6) |
| analogy | 0.397 (4/6) | 0.412 (4/6) |
| morphological | 0.306 (1/6) | 0.335 (2/6) |

## Checks (for the Method section)

**C1. Ranking reliability.** Split-half reliability of the concept scores, splitting the group rankings in two (Spearman-Brown; about 0.7 or more is the usual standard for comparative judgement, Verhavert et al., 2019):

| criterion | brief_id | n_rankings | comparisons_per_concept | split_half |
|---|:---:|:---:|:---:|:---:|
| novelty | A1_kettle | 80 | 16.0 | 0.835 |
| novelty | A2_screwdriver | 80 | 16.0 | 0.856 |
| novelty | A3_toothbrush | 80 | 16.0 | 0.859 |
| novelty | O1_laptopwork | 80 | 16.0 | 0.789 |
| novelty | O2_citytransport | 80 | 16.0 | 0.831 |
| novelty | O3_irrigation | 80 | 16.0 | 0.829 |
| usefulness | A1_kettle | 80 | 16.0 | 0.836 |
| usefulness | A2_screwdriver | 80 | 16.0 | 0.861 |
| usefulness | A3_toothbrush | 80 | 16.0 | 0.828 |
| usefulness | O1_laptopwork | 80 | 16.0 | 0.818 |
| usefulness | O2_citytransport | 80 | 16.0 | 0.837 |
| usefulness | O3_irrigation | 80 | 16.0 | 0.82 |

- Position bias: the concept shown first was ranked top in 21% of groups, the one shown last in 18% (20% = no bias). Mean rank by position shown (position 1: 2.85, position 2: 3.05, position 3: 2.96, position 4: 3.03, position 5: 3.11; equal values = no bias).
- Reversed-order consistency: 96 groups were ranked again with the order reversed. Rankings correlated at rho = 0.82, and 87% of pairs kept the same order. The main analysis uses the first ranking only.

**C3. Length and detail bias.** Correlation of each score with word count (close to 0 means the judge is not rewarding longer or more detailed descriptions):

- novelty: rho with word count = 0.19
- usefulness: rho with word count = -0.07
- mean words per concept: generic 94.5, scamper 93.8, triz 94.5, analogy 94.9, morphological 97.3

- method effects vs generic after adjusting for word count (session means, brief fixed effect):
  - novelty: analogy +1.04 (p = 0.000), morphological +0.59 (p = 0.000), scamper -0.39 (p = 0.002), triz +0.61 (p = 0.000)
  - usefulness: analogy -1.20 (p = 0.000), morphological -0.79 (p = 0.000), scamper -0.39 (p = 0.001), triz -0.98 (p = 0.000)

**C4. Agreement with human experts.** Not done yet. A sheet with 30 pairs is in evaluation_v2/human_check/ (see INSTRUCTIONS.txt).

## Methodology note (for the paper)

Concepts were evaluated by gpt-5.6-luna, a model from a different family to the generator (gemini-3.8-flash), so that no model evaluated its own output. The judge saw the brief and the concepts' titles and descriptions only, never the condition or the method workings. Novelty and usefulness were assessed by comparative judgement (Thurstone, 1927; Pollitt, 2012), which avoids the ceiling effects of absolute ratings and suits LLM judges (Liusie et al., 2024). The judge was shown groups of 5 concepts for the same brief and ranked them from most to least novel, and in a separate call from most to least useful. Each concept appeared in 4 groups drawn at random from all conditions and runs for its brief, in random order; 10% of groups were ranked again in reverse order to measure position bias (Zheng et al., 2023). Rankings were converted to a score per concept with the Plackett-Luce model (Luce, 1959; Plackett, 1975; Hunter, 2004), standardised within each brief, and reliability was estimated by split-half correlation (Verhavert et al., 2019). Method effects are reported as the probability that a method's concept outranks a baseline concept for the same brief, with 95% intervals from resampling sessions. Spread was measured without the judge, as the mean cosine distance between text embeddings of the concepts a method produced for a brief over all runs (text-embedding-3-large; Doshi & Hauser, 2024; Meincke et al., 2024).
