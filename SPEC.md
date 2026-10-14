## 1. Project Overview

**Project Title:** Bayesian Markov Chain Monte Carlo Calibration and Simulation of a Queueing-Based Limit Order Book

The objective is to build a simplified, event-driven limit order book in which buy and sell orders arrive randomly, orders wait in queues, and prices move as those queues evolve. Bayesian inference is used to estimate the unknown event intensities from observed data, while simulation is used to study market dynamics under the estimated parameters.

The project combines queueing theory, continuous-time Markov processes, Bayesian parameter estimation, Markov Chain Monte Carlo (MCMC), and quantitative finance.

## 2. State Representation

The limit order book state at time $t$ is defined as

$$
X_t = (Q_b, Q_a, P_b, P_a),
$$

where:

- $Q_b \geq 1$: queue size at the best bid.
- $Q_a \geq 1$: queue size at the best ask.
- $P_b$: best bid price, measured in integer ticks.
- $P_a$: best ask price, measured in integer ticks.
- $S = P_a - P_b \geq 1$: bid-ask spread.

The state evolves through discrete events occurring in continuous time. The waiting time until the next event is exponentially distributed, conditional on the current state, with a rate equal to the sum of all applicable event intensities.

## 3. Event Dynamics and Intensity Functions

Let $\lambda_e(X_t)$ denote the instantaneous intensity of event type $e$.

| # | Event | Rate | Effect |
|---|---|---|---|
| 1 | Buy limit at best bid | $\exp(\theta_b + \gamma_b \log(1+Q_b))$ | $Q_b \leftarrow Q_b+1$ |
| 2 | Sell limit at best ask | $\exp(\theta_a + \gamma_a \log(1+Q_a))$ | $Q_a \leftarrow Q_a+1$ |
| 3 | Market sell | $\exp(\theta_{mb})$ | $Q_b \leftarrow Q_b-1$ |
| 4 | Market buy | $\exp(\theta_{ma})$ | $Q_a \leftarrow Q_a-1$ |
| 5 | Cancel bid | $\exp(\theta_{cb})Q_b$ | $Q_b \leftarrow Q_b-1$ |
| 6 | Cancel ask | $\exp(\theta_{ca})Q_a$ | $Q_a \leftarrow Q_a-1$ |
| 7 | Buy limit inside spread | $\exp(\theta_{ib})\mathbf{1}[S \geq 2]$ | $P_b \leftarrow P_b+1,\ Q_b \leftarrow 1$ |
| 8 | Sell limit inside spread | $\exp(\theta_{ia})\mathbf{1}[S \geq 2]$ | $P_a \leftarrow P_a-1,\ Q_a \leftarrow 1$ |

Here, $\mathbf{1}[\cdot]$ denotes an indicator function.

The parameters have the following interpretations:

- $\theta_b,\theta_a$: baseline log-intensities of limit-order arrivals.
- $\gamma_b,\gamma_a$: queue-size sensitivity parameters for limit-order arrivals.
- $\theta_{mb},\theta_{ma}$: log-intensities of market orders.
- $\theta_{cb},\theta_{ca}$: log-coefficients of queue-proportional cancellation intensities.
- $\theta_{ib},\theta_{ia}$: log-intensities of inside-spread limit orders.

The queue-dependent limit-order intensities can increase or decrease with queue size depending on the sign of $\gamma$.

Inside-spread orders are allowed only when $S \geq 2$, ensuring that the spread never falls below one tick.

## 4. Queue Depletion and Price Evolution

Queue depletion occurs when a market order or cancellation reduces a queue to zero.

A queue is not allowed to remain at zero. Instead, the corresponding best price moves by one tick, and the depleted queue is replenished by drawing the size of the next price level.

### 4.1 Bid-Side Depletion

If $Q_b$ reaches zero:

1. Decrease the best bid by one tick:

   $$
   P_b \leftarrow P_b-1.
   $$

2. The spread increases by one tick:

   $$
   S \leftarrow S+1.
   $$

3. Redraw the bid queue:

   $$
   Q_b \leftarrow 1+\operatorname{Poisson}(m).
   $$

### 4.2 Ask-Side Depletion

If $Q_a$ reaches zero:

1. Increase the best ask by one tick:

   $$
   P_a \leftarrow P_a+1.
   $$

2. The spread increases by one tick:

   $$
   S \leftarrow S+1.
   $$

3. Redraw the ask queue:

   $$
   Q_a \leftarrow 1+\operatorname{Poisson}(m).
   $$

The replenishment draw represents the size of the queue at the next price level. It must be recorded in the event log.

For inference on the eight event intensities, the replenishment draw is treated as observed and conditioned upon. Its probability is therefore excluded from the event-intensity likelihood.

The parameter $m$ is estimated separately using a Gamma-Poisson conjugate model because it is required for subsequent simulations.

## 5. Nested Model Specification

Four nested models are implemented to progressively capture more realistic market dynamics.

### M0 — Constant-Intensity Model

All event intensities are constant, apart from the exposure restrictions imposed by the spread condition for inside-spread orders.

This model serves as the baseline for parameter estimation and model validation.

### M1 — Queue-Dependent Limit-Order Arrivals

M1 extends M0 by introducing queue-size dependence in the buy and sell limit-order arrival rates through $\gamma_b$ and $\gamma_a$.

The limit-order intensities are

$$
\lambda_b(Q_b)
=
\exp\left(\theta_b+\gamma_b\log(1+Q_b)\right),
$$

$$
\lambda_a(Q_a)
=
\exp\left(\theta_a+\gamma_a\log(1+Q_a)\right).
$$

All other event intensities retain their M0 specifications.

### M2 — Queue-Proportional Cancellation Model

M2 extends M1 by introducing cancellation intensities proportional to the corresponding queue sizes:

$$
\lambda_{cb}(Q_b)=\exp(\theta_{cb})Q_b,
$$

$$
\lambda_{ca}(Q_a)=\exp(\theta_{ca})Q_a.
$$

M2 is the full structural model and contains ten event-intensity parameters:

$$
\Theta =
(
\theta_b,\gamma_b,
\theta_a,\gamma_a,
\theta_{mb},\theta_{ma},
\theta_{cb},\theta_{ca},
\theta_{ib},\theta_{ia}
).
$$

### M3 — Hierarchical Session-Activity Model

M3 extends M2 by introducing a session-level latent activity multiplier that affects all event intensities.

For session $s$,

$$
\eta_s \sim \mathcal{N}(0,\sigma_\eta^2).
$$

Every event intensity in session $s$ is multiplied by

$$
\exp(\eta_s).
$$

Thus, the session-specific intensity is

$$
\lambda_e^{(s)}(X_t)
=
\exp(\eta_s)\lambda_e(X_t;\Theta).
$$

The initial specification uses eight sessions.

The shared latent activity multiplier captures variation in overall market activity between sessions. Because each $\eta_s$ affects all event types in its session, posterior inference couples the event-type parameters through the latent effects.

MCMC is therefore essential for the main joint inference task in M3.

## 6. Starting True Parameters and Simulation Sanity Check

The following values define the initial data-generating configuration.

| Parameter | Initial value |
|---|---:|
| $\theta_b$ | $0.7$ |
| $\gamma_b$ | $-0.3$ |
| $\theta_a$ | $0.6$ |
| $\gamma_a$ | $-0.3$ |
| Market order intensity, each side | $0.8$ |
| Cancellation coefficient, each side | $0.1$ |
| Inside-spread intensity, each side | $1.0$ |
| $m$ | $5$ |

For the market-order intensities,

$$
\exp(\theta_{mb})=\exp(\theta_{ma})=0.8.
$$

For the cancellation coefficients,

$$
\exp(\theta_{cb})=\exp(\theta_{ca})=0.1.
$$

For the inside-spread intensities,

$$
\exp(\theta_{ib})=\exp(\theta_{ia})=1.0.
$$

The simulator must undergo a sanity check before the inference experiments begin.

The initial parameters should be tested and adjusted, if necessary, to achieve the following approximate targets:

- Mean queue size of around 4–6 units on each side.
- Average depletion interval of around 10–30 seconds for each side.
- Sufficient price movements to support meaningful statistical analysis.

These are calibration targets, not guaranteed properties of the initial parameter values. Any adjustments must be documented so that the experiments remain reproducible.

All simulation configurations and random seeds must be recorded.

## 7. Bayesian Priors

For the non-conjugate parameterization, use the following priors:

$$
\theta \sim \mathcal{N}(0,2^2),
$$

$$
\gamma \sim \mathcal{N}(0,1),
$$

$$
\sigma_\eta \sim \operatorname{HalfNormal}(0.5).
$$

The session-level latent activity effects satisfy

$$
\eta_s\mid\sigma_\eta
\sim
\mathcal{N}(0,\sigma_\eta^2).
$$

For validation against exact analytical results, Gamma priors should be used for positive intensity coefficients wherever conjugacy applies.

Use the shape-rate parameterization for Gamma distributions throughout the implementation.

The prior hyperparameters for estimating $m$ should be specified separately in the implementation configuration.

## 8. Event-Intensity Likelihood

Let $N_e(T)$ be the number of observed events of type $e$ in the observation interval $[0,T]$.

For an intensity function $\lambda_e(X_t;\phi_e)$, the event-specific log-likelihood is

$$
\ell_e(\phi_e)
=
\sum_{k:e_k=e}
\log \lambda_e(X_{t_k^-};\phi_e)
-
\int_0^T \lambda_e(X_t;\phi_e)\,dt,
$$

where:

- $e_k$ is the type of event $k$.
- $t_k$ is its event time.
- $X_{t_k^-}$ is the state immediately before that event.
- The integral represents the accumulated intensity over the observation interval.

The total log-likelihood is

$$
\ell(\Theta)
=
\sum_e \ell_e(\phi_e).
$$

The intensity must be evaluated using the state immediately before each event.

The exposure integral must account for all state changes throughout the observed path, including periods when inside-spread events are unavailable because $S<2$.

The realized queue-replenishment draws are conditioned upon and excluded from this likelihood.

## 9. Posterior Factorization and Exact Results

An important theoretical result is that, under independent priors and the specified M0–M2 likelihood, the posterior factorizes across event types.

For M0–M2,

$$
p(\Theta\mid\mathcal{D})
\propto
\prod_e p(\phi_e\mid\mathcal{D}),
$$

where $\mathcal{D}$ denotes the observed event log.

This factorization allows the event-specific posterior distributions to be estimated independently.

### 9.1 Exact Gamma Posteriors

For an event intensity of the form

$$
\lambda_e(X_t)=r_e g_e(X_t),
$$

where $r_e>0$ is an unknown coefficient and $g_e(X_t)$ is a known exposure function, assume

$$
r_e\sim\operatorname{Gamma}(\alpha_e,\beta_e).
$$

Using the shape-rate parameterization, the posterior is

$$
\boxed{
r_e\mid\mathcal{D}
\sim
\operatorname{Gamma}
\left(
\alpha_e+n_e,\,
\beta_e+\int_0^T g_e(X_t)\,dt
\right)
}
$$

where $n_e=N_e(T)$.

The exposure function $g_e(X_t)$ is:

- $g_e(X_t)=1$ for constant-rate events.
- $g_e(X_t)=Q_b$ for bid cancellations.
- $g_e(X_t)=Q_a$ for ask cancellations.
- $g_e(X_t)=\mathbf{1}[S\geq2]$ for inside-spread events.

The exact Gamma posterior provides a benchmark for validating the likelihood and inference implementation.

### 9.2 Non-Conjugate Limit-Order Parameters

The queue-dependent limit-order intensities depend on the pairs

$$
(\theta_b,\gamma_b)
\quad\text{and}\quad
(\theta_a,\gamma_a).
$$

These pairs are non-conjugate under the specified Normal priors.

Each pair defines a two-dimensional posterior that can be evaluated numerically using grid quadrature.

The resulting posterior surfaces should be used to validate MCMC or other numerical inference methods for these parameters.

### 9.3 Why MCMC Is Needed

In M0–M2, the independent-prior and event-specific likelihood structure allows the posterior to be decomposed by event type. Several components have exact Gamma posteriors, while the two-dimensional limit-order posteriors can be checked using numerical grid quadrature.

In M3, the shared session-level latent variable $\eta_s$ multiplies every event intensity within session $s$. This couples all event types through the shared latent activity effects.

Consequently, MCMC is needed for practical joint posterior inference over the structural parameters, session effects, and the hyperparameter $\sigma_\eta$.

## 10. Estimation of the Queue-Replenishment Parameter $m$

Whenever a queue is depleted, the size of the new queue is generated as

$$
Q_{\mathrm{new}}=1+Y,
\qquad
Y\sim\operatorname{Poisson}(m).
$$

For each observed depletion, define

$$
Y_i=Q_{\mathrm{new},i}-1.
$$

Conditional on the observed depletion events, the replenishment observations satisfy

$$
Y_i\overset{\mathrm{iid}}{\sim}\operatorname{Poisson}(m).
$$

Use a Gamma prior

$$
m\sim\operatorname{Gamma}(\alpha_m,\beta_m),
$$

with shape-rate parameters.

If there are $n$ observed replenishment draws and their total is $\sum_{i=1}^n Y_i$, the posterior is

$$
m\mid\{Y_i\}_{i=1}^n
\sim
\operatorname{Gamma}
\left(
\alpha_m+\sum_{i=1}^nY_i,\,
\beta_m+n
\right).
$$

This parameter is estimated separately from the eight event intensities and is used to generate queue sizes during subsequent simulations.

## 11. Misspecification Generator — Experiment 5

Experiment 5 introduces unobserved, time-varying market activity through a two-regime Markov chain.

Let

$$
Z_t\in\{0,1\}
$$

denote the latent regime. The activity multiplier is

$$
a(Z_t)=
\begin{cases}
0.5,&Z_t=0,\\
2.0,&Z_t=1.
\end{cases}
$$

The true event intensities become

$$
\lambda_e^{\mathrm{true}}(X_t)
=
a(Z_t)\lambda_e(X_t;\Theta).
$$

The regime process should be configured to have a mean regime duration of approximately 60 seconds.

The regime is unobserved by the fitted models. The experiment therefore measures the effects of unmodeled market-activity fluctuations on parameter recovery, posterior uncertainty, and predictive performance.

The transition probabilities of the regime chain must be specified in the experiment configuration and documented in the results.

## 12. Event Logging and Data Format

Every simulated event must produce a log record with the following columns, in the specified order:

`t, event, Qb_before, Qa_before, Pb_before, Pa_before, session, Qb_after, Qa_after, Pb_after, Pa_after`

Column definitions:

| Column | Description |
|---|---|
| `t` | Cumulative simulation time |
| `event` | Event type identifier |
| `Qb_before` | Bid queue size immediately before the event |
| `Qa_before` | Ask queue size immediately before the event |
| `Pb_before` | Best bid price immediately before the event |
| `Pa_before` | Best ask price immediately before the event |
| `session` | Session identifier |
| `Qb_after` | Bid queue size after the complete transition |
| `Qa_after` | Ask queue size after the complete transition |
| `Pb_after` | Best bid price after the complete transition |
| `Pa_after` | Best ask price after the complete transition |

The `after` fields must represent the state after all effects of the event have been applied, including price changes and queue replenishment caused by depletion.

The replenishment draw must also be recorded whenever depletion occurs. This may be implemented using additional columns or a separate event-metadata file without changing the required core columns.

The event log must preserve chronological order and contain sufficient information to reconstruct the observed state path.

## 13. Validation and Experiments

The implementation must support the following experiments.

### Experiment 1 — Simulator Sanity Check

- Verify event-rate calculations.
- Verify continuous-time waiting-time generation and event selection.
- Verify that completed transitions preserve $Q_b,Q_a\geq1$.
- Verify that the spread remains $S\geq1$.
- Verify the depletion, price movement, and replenishment rules.
- Measure mean queue sizes, depletion intervals, and price movements.
- Adjust and document simulation parameters if necessary to reach the sanity-check targets.

### Experiment 2 — Constant-Rate Parameter Recovery

- Generate synthetic data under M0.
- Fit M0 to the simulated observations.
- Compare posterior estimates with known generating parameters.
- Validate the exact Gamma posterior where applicable.

### Experiment 3 — Queue-Dependent and Cancellation Parameter Recovery

- Generate synthetic data under M1 and M2.
- Estimate queue-dependent limit-order and cancellation parameters.
- Compare the two-dimensional limit-order posteriors against grid quadrature.
- Evaluate parameter recovery as the observation duration and event counts change.

### Experiment 4 — Hierarchical Session-Activity Model

- Generate data from eight sessions with session-level latent activity.
- Fit M3 using MCMC.
- Estimate structural parameters, session effects, and $\sigma_\eta$.
- Evaluate convergence, effective sample size, posterior uncertainty, and parameter recovery.

### Experiment 5 — Model Misspecification

- Generate observations under the two-regime activity-modulated process.
- Fit models that do not observe the latent regime.
- Measure parameter bias, uncertainty calibration, residual patterns, and predictive performance.
- Discuss the limitations of inference under model misspecification.

### Experiment 6 — Posterior Predictive Simulation

- Draw model parameters from their fitted posterior distributions.
- Simulate new limit order book trajectories.
- Compare queue-size distributions, depletion intervals, spread distributions, and price movements against held-out observations or reference simulations.
- Assess uncertainty across posterior predictive trajectories.

## 14. Evaluation Metrics

The project should report:

- Parameter estimation error and bias.
- Credible interval coverage.
- Posterior uncertainty and calibration.
- MCMC convergence diagnostics, including trace plots, effective sample sizes, and $\hat{R}$.
- Mean queue sizes and queue-size distributions.
- Bid-side and ask-side depletion frequencies and intervals.
- Spread distributions and price-change statistics.
- Posterior predictive discrepancies.
- Computational runtime and scalability.

Parameter-recovery results must be distinguished from predictive-performance results.

## 15. Repository Structure

The repository should include the following components:

```text
Queueing-Based-Limit-Order-Book/
├── README.md
├── SPEC.md
├── Makefile
├── requirements.txt
├── configs/
│   └── default.yaml
├── src/
│   ├── simulator.py
│   ├── likelihood.py
│   ├── inference.py
│   ├── diagnostics.py
│   └── posterior_predictive.py
├── tests/
│   ├── test_simulator.py
│   ├── test_likelihood.py
│   ├── test_conjugate.py
│   └── test_inference.py
├── experiments/
│   ├── experiment_1.py
│   ├── experiment_2.py
│   ├── experiment_3.py
│   ├── experiment_4.py
│   └── experiment_5.py
├── data/
│   └── .gitkeep
├── results/
│   └── .gitkeep
└── notebooks/
    └── exploration.ipynb
```

The precise module boundaries may evolve during development, but simulation, likelihood evaluation, inference, diagnostics, and experiment orchestration should remain logically separated.

Large generated datasets and experiment outputs should not be committed unless they are small, reproducible examples needed for documentation.

## 16. Engineering and Reproducibility Requirements

- Use explicit random seeds for all simulations and experiments.
- Keep model parameters and important simulation settings in configuration files.
- Separate data generation from parameter estimation.
- Include automated tests for state transitions, likelihood calculations, and posterior computations.
- Provide Makefile targets for testing and running standard experiments.
- Document dependencies and environment setup.
- Use numerically stable log-intensity and log-posterior calculations.
- Record the model version and configuration alongside experiment outputs.
- Ensure that the simulation and inference pipelines can be reproduced from the saved configuration and random seed.

## 17. Expected Deliverables

The completed project should provide:

1. A reproducible continuous-time limit order book simulator.
2. A validated event-intensity likelihood implementation.
3. Exact conjugate posterior calculations for suitable event types.
4. Grid-based validation of the non-conjugate two-dimensional limit-order posteriors.
5. An MCMC implementation for the hierarchical session-activity model.
6. A two-regime misspecification experiment.
7. Posterior predictive simulations and quantitative evaluation.
8. Automated tests and reproducible experiment configurations.
9. A technical report discussing assumptions, inference accuracy, model limitations, and implications for quantitative finance.

## 18. Scope and Limitations

This project develops a simplified queueing-based market model for statistical experimentation.

It does not attempt to reproduce every feature of a real exchange, including:

- Multiple levels of order-book depth.
- Hidden liquidity and iceberg orders.
- Strategic agent behaviour.
- Exchange-specific matching rules.
- Additional order-flow dependencies beyond the specified intensities.

The results should be interpreted as a study of stochastic market dynamics and Bayesian parameter estimation rather than as a calibrated production trading system.
