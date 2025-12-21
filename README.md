# PhenTAX: a JAX Implementation of the IMRPhenomT(HM) waveform family

`phentax` is a JAX re-implementation of the IMRPhenomT and IMRPhenomTHM gravitational waveform models as implemented in the [`phenomxpy` package](https://gitlab.com/imrphenom-dev/phenomxpy).

At present, `phentax` supports the generation of time-domain waveforms for quasi-circular, non-precessing binary black hole mergers, including higher-order modes (HMs) in the IMRPhenomTHM model. The implementation leverages JAX's capabilities for automatic differentiation, just-in-time compilation, and batching to enable efficient waveform generation and parameter estimation.

## Installation
This project is managed by [uv](https://docs.astral.sh/uv/). To set up the development environment, first clone the repository:
```
git clone #<repository_url>
cd phentax
uv install
```
This will create a virtual environment and install all necessary dependencies. Run your commands within this environment using
```
uv run <YOUR-COMMAND>
```
