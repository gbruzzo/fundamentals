# `docs/chapters/` — per-chapter overviews

One page per book chapter. Each page maps the book's exposition to the
orchestrator scripts in `chapters/chapter_<N>/` and to the library
components in `src/active_inference/` they exercise.

## Pages

| File | Mirrors | Purpose |
|---|---|---|
| [`chapter_01.md`](chapter_01.md) | Book Ch. 1 — *The Hypothesis-Testing Brain* | Box scenario, three perspectives, Bayes' theorem walk-through, inverse problem demos. |
| [`chapter_02.md`](chapter_02.md) | Book Ch. 2 — *Hidden State Estimation* | The full linear-Gaussian recipe + Examples 2.1–2.10. |
| [`chapter_03.md`](chapter_03.md) | Book Ch. 3 — *Combining Learning and Inference* | OLS / GD / Bayesian linear regression, multivariate Gaussians, LGS, factor-analysis EM. |
| [`chapter_04.md`](chapter_04.md) | Book Ch. 4 — *Variational Bayesian Inference* | Variational free energy & its five forms, coordinate search, fixed-form VI, mean-field CAVI. |
| [`chapter_05.md`](chapter_05.md) | Book Ch. 5 — *Predictive Coding* | Prediction-error form of VFE, recognition dynamics (Eq. 16), univariate / multivariate / hierarchical PC, the unified Ch.4–5 visualization layer. |
| [`chapter_06.md`](chapter_06.md) | Book Ch. 6 — *Generalized Filtering for Perception* (Part II) | Dynamic state-space model, generalized coordinates, correlated embedding-order precision, and Example 6.7 vector filtering. |
| [`chapter_07.md`](chapter_07.md) | Book Ch. 7 — *Active Generalized Filtering* (Part II) | Action + preference prior, action via the forward model, homeostatic set-point regulation, and §7.5 multivariate AIF in generalized coordinates. |
| [`chapter_08.md`](chapter_08.md) | Book Ch. 8 — *Learning, attention, and hierarchical models* (Part II) | Continuous-state triple estimation: first-order parameter learning, second-order log-precision attention, hierarchy, and message passing. |
| [`chapter_09.md`](chapter_09.md) | Book Ch. 9 — *Active Inference in POMDPs* (Part II) | Discrete/categorical formulation: the `A`/`B`/`C`/`D` matrices and exact hidden-state inference (`s = σ(log Aᵀô + log D)`). |
| [`chapter_10.md`](chapter_10.md) | Book Ch. 10 — *Learning & extensions in POMDPs* (Part II) | §10.1 Dirichlet learning of `A`/`B`/`D` + parameter novelty; §10.2 habit `E` + policy precision `γ`; §10.3 factorial depth (two-armed bandit); §10.4 hierarchical depth. |

## How to read a chapter page

Each page has the same skeleton:

1. *What the chapter covers* — a paragraph summary in our own words.
2. *Recipe* — the canonical 5-step modeling pattern, parameterized for
   that chapter.
3. *Scripts* — table mapping each orchestrator to the question it
   answers and the library symbols it imports.
4. *Where the book takes this next* — a forward pointer to the next
   chapter's chapter-page in this folder.

## See also

- [`../topics/`](../topics/) — concept-by-concept walkthroughs that span
  more than one chapter.
- [`../reference/`](../reference/) — full API listing for the subpackages
  the chapter scripts import from.
