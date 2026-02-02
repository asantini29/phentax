# PhenTAX: a JAX Implementation of the IMRPhenomTHM waveform family

[![Documentation](https://github.com/asantini29/phentax/actions/workflows/docs.yml/badge.svg)](https://asantini29.github.io/phentax/)

`phentax` is a JAX re-implementation of the IMRPhenomT and IMRPhenomTHM gravitational waveform models as implemented in the [`phenomxpy` package](https://gitlab.com/imrphenom-dev/phenomxpy).

At present, `phentax` supports the generation of time-domain waveforms for quasi-circular, non-precessing binary black hole mergers, including higher-order modes (HMs) in the IMRPhenomTHM model. The implementation leverages JAX's capabilities for automatic differentiation, just-in-time compilation, and batching to enable efficient waveform generation and parameter estimation.

## Installation
This project is managed by [uv](https://docs.astral.sh/uv/). To set up the development environment, first clone the repository:
```
git clone https://github.com/asantini29/phentax
cd phentax
```

### GPU Installation (CUDA)
For systems with NVIDIA GPUs and CUDA support, install with:
```
uv sync --group gpu
```

### CPU Installation
For CPU-only systems, install with:
```
uv sync --group cpu
```

This will create a virtual environment and install all necessary dependencies. Run your commands within this environment using
```
uv run <YOUR-COMMAND>
```

## Credits
The original `python` implementation of the IMRPhenomT(HM) waveform models in the `phenomxpy` package were developed by Cecilio García Quirós. This JAX re-implementation builds upon his work. If you use this code in your research, please cite both the original `phenomxpy` package and this `phentax` implementation:
```
@misc{garcíaquirós2025gpuacceleratedlisaparameterestimation,
      title={GPU-accelerated LISA parameter estimation with full time domain response},
      author={Cecilio García-Quirós and Shubhanshu Tiwari and Stanislav Babak},
      year={2025},
      eprint={2501.08261},
      archivePrefix={arXiv},
      primaryClass={gr-qc},
      url={https://arxiv.org/abs/2501.08261},
}
```
