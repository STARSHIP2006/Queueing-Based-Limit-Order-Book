# Queueing-Based-Limit-Order-Book

Bayesian Markov Chain Monte Carlo Calibration and Simulation of a Queueing-Based Limit Order Book

## Project Overview

This project develops a simplified, event-driven limit order book simulator and uses Bayesian inference to estimate its underlying event intensities. It combines queueing theory, continuous-time Markov processes, Bayesian parameter estimation, and Markov Chain Monte Carlo (MCMC) to study stochastic market dynamics.

## State Representation

The limit order book state is defined as

$$
X_t = (Q_b, Q_a, P_b, P_a),
$$

where $Q_b$ and $Q_a$ denote the best bid and ask queue sizes, $P_b$ and $P_a$ denote their respective prices, and the bid-ask spread is $S = P_a - P_b \geq 1$.

## Event Dynamics

The simulator models eight types of events:

- Buy and sell limit-order arrivals.
- Market buy and market sell orders.
- Bid-side and ask-side cancellations.
- Buy and sell limit orders placed inside the spread.

Event intensities depend on queue sizes, market conditions, and model parameters. Queue depletion triggers price movements and stochastic replenishment of the depleted queue.

## Bayesian Calibration

The project implements four nested models:

- **M0:** Constant event intensities.
- **M1:** Queue-dependent limit-order arrival intensities.
- **M2:** Queue-proportional cancellation intensities.
- **M3:** Session-level latent activity multipliers with hierarchical Bayesian inference.

Conjugate Gamma posteriors and numerical grid quadrature provide validation benchmarks, while MCMC enables inference under the hierarchical model.

## Experiments and Evaluation

The project investigates parameter recovery, posterior uncertainty, model misspecification, and posterior predictive performance. A two-regime Markov process is introduced to study the effects of unobserved changes in market activity.

Performance is evaluated using parameter estimation error, credible interval coverage, MCMC convergence diagnostics, queue-size distributions, depletion frequencies, spread dynamics, and simulated price movements.

## Technologies

- Python
- NumPy, pandas, SciPy
- Matplotlib
- Bayesian inference and MCMC
- Queueing theory and stochastic processes

## Objective

To build a reproducible stochastic market simulator, develop and validate Bayesian calibration methods, and analyze how queue dynamics and latent market activity influence limit order book behavior.
