## 1. Project Overview

**Project Title:** Bayesian MCMC Calibration, Kalman Filtering, and Simulation of a Queueing-Based Limit Order Book

The objective is to build a simplified, event-driven limit order book in which buy and sell orders arrive randomly, orders wait in queues, and prices move as those queues evolve. Bayesian inference is used to estimate the unknown event intensities from observed data, while simulation is used to study market dynamics under the estimated parameters.

The project combines queueing theory, continuous-time Markov processes, Bayesian parameter estimation, Markov Chain Monte Carlo (MCMC), Kalman filtering, and quantitative finance.

The project uses one full structural model for the order-book event intensities. A separate Kalman-filter experiment estimates a time-varying market-activity factor from aggregated event counts; it does not introduce a separate session-level hierarchical model.

## 2. State Representation

The limit order book state at time $t$ is defined as

```math
X_t = (Q_b, Q_a, P_b, P_a),
```

where:

- $Q_b \geq 1$: queue size at the best bid.
- $Q_a \geq 1$: queue size at the best ask.
- $P_b$: best bid price, measured in integer ticks.
- $P_a$: best ask price, measured in integer ticks.
- $S = P_a - P_b \geq 1$: bid-ask spread.

The state evolves through discrete events occurring in continuous time. Conditional on the current state, the waiting time until the next event is exponentially distributed, with rate equal to the sum of all applicable event intensities.

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

The queue-dependent limit-order intensities can increase or decrease with queue size depending on the sign of $\gamma$. The cancellation intensities are proportional to queue size, corresponding to the assumption that each resting order has the same independent cancellation hazard.

Inside-spread orders are allowed only when $S \geq 2$, ensuring that the spread never falls below one tick.

## 4. Queue Depletion and Price Evolution

Queue depletion occurs when a market order or cancellation reduces a queue to zero.

A queue is not allowed to remain at zero. Instead, the corresponding best price moves by one tick, and the depleted queue is replenished by drawing the size of the next price level.

### 4.1 Bid-Side Depletion

If $Q_b$ reaches zero:

1. Decrease the best bid by one tick: $P_b \leftarrow P_b-1$.
2. The spread increases by one tick: $S \leftarrow S+1$.
3. Redraw the bid queue: $Q_b \leftarrow 1+\mathrm{Poisson}(m)$.

### 4.2 Ask-Side Depletion

If $Q_a$ reaches zero:

1. Increase the best ask by one tick: $P_a \leftarrow P_a+1$.
2. The spread increases by one tick: $S \leftarrow S+1$.
3. Redraw the ask queue: $Q_a \leftarrow 1+\mathrm{Poisson}(m)$.

The replenishment draw represents the size of the queue at the next price level. It must be recorded in the event log.

For inference on the eight event intensities, the replenishment draw is treated as observed and conditioned upon. Its probability is therefore excluded from the event-intensity likelihood.

The parameter $m$ is estimated separately using a Gamma-Poisson conjugate model because it is required for subsequent simulations.

## 5. Full Structural Intensity Model

The project uses a single structural model containing all eight event types specified in Section 3. The model includes both queue-dependent limit-order arrival rates and queue-proportional cancellation rates; these components are fitted together rather than presented as separate nested models.

The queue-dependent limit-order intensities are

```math
\lambda_b(Q_b)=\exp\left(\theta_b+\gamma_b\log(1+Q_b)\right),
```

```math
\lambda_a(Q_a)=\exp\left(\theta_a+\gamma_a\log(1+Q_a)\right).
```

The cancellation intensities are

```math
\lambda_{cb}(Q_b)=\exp(\theta_{cb})Q_b,
```

```math
\lambda_{ca}(Q_a)=\exp(\theta_{ca})Q_a.
```

The complete event-intensity parameter vector contains ten parameters:

```math
\Theta = (\theta_b,\gamma_b,\theta_a,\gamma_a,\theta_{mb},\theta_{ma},\theta_{cb},\theta_{ca},\theta_{ib},\theta_{ia}).
```

The inside-spread intensities are additionally gated by $\mathbf{1}[S\geq2]$. All event intensities are recomputed after each event changes the state.

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

```math
\exp(\theta_{mb})=\exp(\theta_{ma})=0.8.
```

For the cancellation coefficients,

```math
\exp(\theta_{cb})=\exp(\theta_{ca})=0.1.
```

For the inside-spread intensities,

```math
\exp(\theta_{ib})=\exp(\theta_{ia})=1.0.
```

The simulator must undergo a sanity check before the inference experiments begin.

The initial parameters should be tested and adjusted, if necessary, to achieve the following approximate targets:

- Mean queue size of around 4-6 units on each side.
- Average depletion interval of around 10-30 seconds for each side.
- Sufficient price movements to support meaningful statistical analysis.

These are calibration targets, not guaranteed properties of the initial parameter values. Any adjustments must be documented so that the experiments remain reproducible. All simulation configurations and random seeds must be recorded.

## 7. Bayesian Priors

For the primary non-conjugate parameterization, use independent priors

```math
\theta_j\sim\mathcal{N}(0,2^2)
```

for each log-intensity parameter $\theta_j$, and

```math
\gamma_b,\gamma_a\sim\mathcal{N}(0,1).
```

For validation against exact analytical results, use Gamma priors on positive rate coefficients wherever conjugacy applies. Use the shape-rate parameterization for Gamma distributions throughout the implementation.

The prior hyperparameters for estimating $m$ must be specified separately in the implementation configuration.

## 8. Event-Intensity Likelihood

Let $N_e(T)$ be the number of observed events of type $e$ in the observation interval $[0,T]$.

For an intensity function $\lambda_e(X_t;\phi_e)$, the event-specific log-likelihood is

```math
\ell_e(\phi_e)=\sum_{k:e_k=e}\log\lambda_e(X_{t_k^-};\phi_e)-\int_0^T\lambda_e(X_t;\phi_e)\,dt,
```

where:

- $e_k$ is the type of event $k$.
- $t_k$ is its event time.
- $X_{t_k^-}$ is the state immediately before that event.
- The integral represents the accumulated intensity over the observation interval.

The total log-likelihood is

```math
\ell(\Theta)=\sum_e\ell_e(\phi_e).
```

The intensity must be evaluated using the state immediately before each event. The exposure integral must account for all state changes throughout the observed path, including periods when inside-spread events are unavailable because $S<2$.

The realized queue-replenishment draws are conditioned upon and excluded from this likelihood.

## 9. Bayesian Inference and Analytical Validation

Under independent priors and the specified structural intensity functions, the likelihood contributions for distinct event types separate when conditioned on the observed state path. The two parameters for buy limit-order arrivals, $(\theta_b,\gamma_b)$, must be inferred jointly; similarly, $(\theta_a,\gamma_a)$ must be inferred jointly. The other event types each have a single rate coefficient.

### 9.1 Exact Gamma Posteriors

For an event intensity of the form

```math
\lambda_e(X_t)=r_e g_e(X_t),
```

where $r_e>0$ is an unknown coefficient and $g_e(X_t)$ is a known exposure function, assume

```math
r_e\sim\mathrm{Gamma}(\alpha_e,\beta_e).
```

Using the shape-rate parameterization, the posterior is

```math
r_e\mid\mathcal{D}\sim\mathrm{Gamma}\left(\alpha_e+n_e,\ \beta_e+\int_0^T g_e(X_t)\,dt\right),
```

where $n_e=N_e(T)$ and $\mathcal{D}$ denotes the observed event log.

The exposure function $g_e(X_t)$ is:

- $g_e(X_t)=1$ for constant-rate events.
- $g_e(X_t)=Q_b$ for bid cancellations.
- $g_e(X_t)=Q_a$ for ask cancellations.
- $g_e(X_t)=\mathbf{1}[S\geq2]$ for inside-spread events.

For this analytical validation, assign Gamma priors directly to the positive rate coefficients, such as $r_e$ or cancellation coefficients $c_b=\exp(\theta_{cb})$ and $c_a=\exp(\theta_{ca})$. This is a separate conjugate parameterization used to verify the likelihood and posterior calculations; it is not the same as placing Normal priors on their logarithms.

### 9.2 Non-Conjugate Limit-Order Parameters

The queue-dependent limit-order rates depend on the pairs $(\theta_b,\gamma_b)$ and $(\theta_a,\gamma_a)$. These pairs are non-conjugate under the specified Normal priors.

Each pair defines a two-dimensional posterior that can be evaluated numerically using grid quadrature. MCMC should also be implemented to draw posterior samples from these distributions and compared with the grid-quadrature benchmark. This provides a direct validation of the sampling implementation even though the parameter pairs are only two-dimensional.

### 9.3 Role of MCMC

MCMC is used for posterior sampling of the non-conjugate queue-dependent limit-order parameters. The scalar rate coefficients can be checked against exact Gamma posteriors in the conjugate validation setup. Grid quadrature provides an independent benchmark for assessing MCMC accuracy, mixing, and credible intervals.

## 10. Estimation of the Queue-Replenishment Parameter $m$

Whenever a queue is depleted, the size of the new queue is generated as

```math
Q_{\mathrm{new}}=1+Y,\qquad Y\sim\mathrm{Poisson}(m).
```

For each observed depletion, define

```math
Y_i=Q_{\mathrm{new},i}-1.
```

Conditional on the observed depletion events, the replenishment observations satisfy

```math
Y_i\overset{\mathrm{iid}}{\sim}\mathrm{Poisson}(m).
```

Use a Gamma prior

```math
m\sim\mathrm{Gamma}(\alpha_m,\beta_m),
```

with shape-rate parameters. If there are $n$ observed replenishment draws and their total is $\sum_{i=1}^n Y_i$, the posterior is

```math
m\mid\{Y_i\}_{i=1}^n\sim\mathrm{Gamma}\left(\alpha_m+\sum_{i=1}^nY_i,\ \beta_m+n\right).
```

This parameter is estimated separately from the eight event intensities and is used to generate queue sizes during subsequent simulations.

## 11. Kalman Filter for Time-Varying Market Activity

This experiment extends the simulator with a time-varying activity factor for the purpose of testing a Kalman filter. It does not replace the core structural model or introduce a session-level hierarchical model.

### 11.1 Latent Activity State

Divide the simulated or observed event log into fixed-width time bins indexed by $k$. Let $h_k$ represent the latent log-activity level during bin $k$. The event intensities in this experiment are

```math
\lambda_{e,k}(X_t)=\exp(h_k)\lambda_e^0(X_t;\Theta),
```

where $\lambda_e^0(X_t;\Theta)$ is the baseline intensity from Sections 3 and 5, and $\exp(h_k)$ is a common activity multiplier applied to all event types during bin $k$.

Model the latent activity as a mean-reverting Gaussian state process:

```math
h_k=\rho h_{k-1}+w_k,\qquad w_k\sim\mathcal{N}(0,q),\qquad |\rho|<1.
```

Here, $\rho$ controls persistence and $q$ is the state-noise variance. The parameters $\rho$ and $q$, the time-bin width, and the initial state distribution must be recorded in the experiment configuration.

### 11.2 Constructing the Observation

Let $N_k$ be the total number of observed events in time bin $k$. Using the calibrated baseline parameters and the observed state path, calculate the baseline integrated intensity

```math
A_k=\int_{t_{k-1}}^{t_k}\sum_e\lambda_e^0(X_t;\widehat{\Theta})\,dt.
```

An approximate count observation model is

```math
N_k\mid h_k\approx\mathrm{Poisson}(A_k\exp(h_k)).
```

To use a standard linear Kalman filter, transform the count into a log-rate observation:

```math
z_k=\log\left(\frac{N_k+0.5}{A_k}\right).
```

For bins with positive baseline exposure, use the approximate linear observation equation

```math
z_k\approx h_k+v_k,\qquad v_k\sim\mathcal{N}(0,R_k),
```

with a delta-method observation variance such as

```math
R_k\approx\frac{1}{N_k+0.5}.
```

The Poisson count model and its transformed Gaussian observation are approximations for filtering purposes, not an exact linear-Gaussian representation of the original event process. The experiment must assess sensitivity to the bin width, especially when bins contain few events. Bins with zero or invalid baseline exposure must be handled explicitly rather than passed into the logarithm.

### 11.3 Kalman Filter Equations

Let $\widehat{h}_{k\mid k}$ and $P_{k\mid k}$ be the filtered mean and variance after observing bin $k$. The prediction step is

```math
\widehat{h}_{k\mid k-1}=\rho\widehat{h}_{k-1\mid k-1},
```

```math
P_{k\mid k-1}=\rho^2P_{k-1\mid k-1}+q.
```

The Kalman gain is

```math
K_k=\frac{P_{k\mid k-1}}{P_{k\mid k-1}+R_k}.
```

The update step is

```math
\widehat{h}_{k\mid k}=\widehat{h}_{k\mid k-1}+K_k\left(z_k-\widehat{h}_{k\mid k-1}\right),
```

```math
P_{k\mid k}=(1-K_k)P_{k\mid k-1}.
```

The estimated activity multiplier is $\exp(\widehat{h}_{k\mid k})$. This multiplier can be used to adjust the baseline event intensities for online activity tracking or to compare the filtered activity estimate with the known activity path in a synthetic experiment.

### 11.4 Evaluation

Generate synthetic data with a known time-varying activity path. Compare the filtered estimate $\widehat{h}_{k\mid k}$ and multiplier $\exp(\widehat{h}_{k\mid k})$ against the true values. Evaluate tracking error, lag after activity changes, interval calibration based on $P_{k\mid k}$, and sensitivity to bin width and observation noise.

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
| `session` | Session identifier; retained for data organization only |
| `Qb_after` | Bid queue size after the complete transition |
| `Qa_after` | Ask queue size after the complete transition |
| `Pb_after` | Best bid price after the complete transition |
| `Pa_after` | Best ask price after the complete transition |

The `after` fields must represent the state after all effects of the event have been applied, including price changes and queue replenishment caused by depletion.

The replenishment draw must also be recorded whenever depletion occurs. This may be implemented using additional columns or a separate event-metadata file without changing the required core columns.

For synthetic Kalman-filter experiments, the true activity state and any synthetic activity-regime label may be stored in an additional experiment metadata file. These truth values must not be used as filter observations; they are reserved for evaluation.

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

### Experiment 2 — Full Structural Parameter Recovery

- Generate synthetic data using the full structural model in Section 5.
- Fit the ten event-intensity parameters to the simulated observations.
- Compare posterior estimates and credible intervals with the known generating values.
- Evaluate parameter recovery under different observation durations and event counts.

### Experiment 3 — Conjugate and Grid-Quadrature Validation

- Validate the exact Gamma posterior for constant-rate, cancellation, inside-spread, and queue-replenishment coefficients under the conjugate prior setup.
- Compute grid-based posterior reference distributions for the two queue-dependent limit-order parameter pairs.
- Compare MCMC samples, posterior summaries, and credible intervals with the grid-quadrature results.

### Experiment 4 — Kalman Filtering of Time-Varying Activity

- Generate data using a known, smoothly varying latent log-activity process.
- Estimate the baseline structural parameters or use known parameters in a controlled simulation, clearly documenting which setting is used.
- Aggregate event data into fixed-width bins and construct the log-rate observations defined in Section 11.
- Run the Kalman filter to estimate the latent activity level and activity multiplier over time.
- Compare filtered estimates with the true latent activity path using RMSE, tracking lag, and uncertainty diagnostics.
- Repeat for multiple bin widths to assess the count-to-Gaussian approximation.

### Experiment 5 — Kalman Filter Stress Test Under Abrupt Activity Changes

- Generate synthetic data in which the activity multiplier switches between $0.5$ and $2.0$, with a mean regime duration of approximately 60 seconds.
- Do not reveal the true regime labels to the filter; retain them only for evaluation.
- Apply the Kalman filter from Section 11 to the generated event counts.
- Measure how quickly the filter responds to regime changes, whether it oversmooths abrupt transitions, and how tracking accuracy depends on process-noise variance and bin width.
- Explain that a Gaussian state model is a smooth-tracking approximation to abrupt regime changes, so lag during transitions is an expected limitation rather than a coding error.

### Experiment 6 — Posterior Predictive Simulation

- Draw structural parameters from the fitted posterior distributions.
- Simulate new limit order book trajectories.
- Compare queue-size distributions, depletion intervals, spread distributions, and price movements against held-out observations or reference simulations.
- Where appropriate, compare simulations using baseline rates with simulations adjusted by the filtered activity multiplier.
- Assess uncertainty across posterior predictive trajectories.

## 14. Evaluation Metrics

The project should report:

- Parameter estimation error and bias.
- Credible interval coverage.
- Posterior uncertainty and calibration.
- MCMC trace plots, effective sample sizes, and convergence diagnostics such as $\hat{R}$.
- Agreement between MCMC posterior summaries and grid-quadrature benchmarks.
- Mean queue sizes and queue-size distributions.
- Bid-side and ask-side depletion frequencies and intervals.
- Spread distributions and price-change statistics.
- Kalman-filter activity RMSE, tracking lag, and uncertainty calibration.
- Sensitivity of Kalman-filter performance to bin width and process-noise variance.
- Posterior predictive discrepancies.
- Computational runtime and scalability.

Parameter-recovery results must be distinguished from predictive-performance and activity-tracking results.

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
│   ├── kalman_filter.py
│   ├── diagnostics.py
│   └── posterior_predictive.py
├── tests/
│   ├── test_simulator.py
│   ├── test_likelihood.py
│   ├── test_conjugate.py
│   ├── test_inference.py
│   └── test_kalman_filter.py
├── experiments/
│   ├── experiment_1.py
│   ├── experiment_2.py
│   ├── experiment_3.py
│   ├── experiment_4.py
│   ├── experiment_5.py
│   └── experiment_6.py
├── data/
│   └── .gitkeep
├── results/
│   └── .gitkeep
└── notebooks/
    └── exploration.ipynb
```

The precise module boundaries may evolve during development, but simulation, likelihood evaluation, inference, Kalman filtering, diagnostics, and experiment orchestration should remain logically separated.

Large generated datasets and experiment outputs should not be committed unless they are small, reproducible examples needed for documentation.

## 16. Engineering and Reproducibility Requirements

- Use explicit random seeds for all simulations and experiments.
- Keep model parameters and important simulation settings in configuration files.
- Separate data generation from parameter estimation.
- Include automated tests for state transitions, likelihood calculations, conjugate posteriors, MCMC inference, and Kalman-filter prediction/update steps.
- Provide Makefile targets for testing and running standard experiments.
- Document dependencies and environment setup.
- Use numerically stable log-intensity and log-posterior calculations.
- Record the model version and configuration alongside experiment outputs.
- Ensure that the simulation and inference pipelines can be reproduced from the saved configuration and random seed.
- Document the approximations used by the Kalman filter, including the count transformation, observation variance, and bin width.

## 17. Expected Deliverables

The completed project should provide:

1. A reproducible continuous-time limit order book simulator.
2. A validated event-intensity likelihood implementation for the full structural model.
3. Exact conjugate posterior calculations for suitable positive rate coefficients and for $m$.
4. Grid-based validation of the non-conjugate queue-dependent limit-order posteriors.
5. An MCMC implementation with diagnostics and comparison against grid quadrature.
6. A Kalman filter for tracking a time-varying latent market-activity factor from binned event counts.
7. A stress test showing how the filter responds to abrupt changes in market activity.
8. Posterior predictive simulations and quantitative evaluation.
9. Automated tests and reproducible experiment configurations.
10. A technical report discussing assumptions, inference accuracy, Kalman-filter approximation limits, and implications for quantitative finance.

## 18. Scope and Limitations

This project develops a simplified queueing-based market model for statistical experimentation.

It does not attempt to reproduce every feature of a real exchange, including:

- Multiple levels of order-book depth.
- Hidden liquidity and iceberg orders.
- Strategic agent behaviour.
- Exchange-specific matching rules.
- Additional order-flow dependencies beyond the specified intensities.

The Kalman filter operates on aggregated, transformed event counts and uses an approximate Gaussian observation model. It is designed to track broad changes in market activity, not to provide an exact filter for the underlying Poisson event process. Performance may deteriorate when bins contain very few events or when activity changes abruptly.

<<<<<<< HEAD
The results should be interpreted as a study of stochastic market dynamics, Bayesian parameter estimation, and approximate latent-state tracking rather than as a calibrated production trading system.
=======
The results should be interpreted as a study of stochastic market dynamics, Bayesian parameter estimation, and approximate latent-state tracking rather than as a calibrated production trading system.
>>>>>>> 7b69e6dbfc4f2100a133dbb83f50c1bce9328488
