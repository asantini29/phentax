# PhenTAX: a JAX Implementation of the IMRPhenomTHM waveform family

[![Documentation](https://github.com/asantini29/phentax/actions/workflows/docs.yml/badge.svg)](https://asantini29.github.io/phentax/)
[![TestPyPI](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Ftest.pypi.org%2Fpypi%2Fphentax%2Fjson&query=%24.info.version&label=TestPyPI&logo=pypi)](https://test.pypi.org/project/phentax/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/asantini29/phentax/blob/main/LICENSE)

`phentax` is a JAX re-implementation of the IMRPhenomT and IMRPhenomTHM gravitational waveform models as implemented in the [`phenomxpy` package](https://gitlab.com/imrphenom-dev/phenomxpy).

At present, `phentax` supports the generation of time-domain waveforms for quasi-circular, non-precessing binary black hole mergers, including higher-order modes (HMs) in the IMRPhenomTHM model. The implementation leverages JAX's capabilities for automatic differentiation, just-in-time compilation, and batching to enable efficient waveform generation and parameter estimation.

## Note
This package is currently released as a beta version, and it could still show some undesired behaviors. If you encounter any issues, please report them on the [GitHub issues tracker](https://github.com/asantini29/phentax/issues). We welcome any contribution to the codebase: if you want to contribute, please open a pull request or contact the authors directly.

At present, publications using this code are discouraged until the package reaches a stable release. Please contact the authors if you want to discuss this further.

## Installation

### Installation from Package Index
`phentax` can be installed directly with `pip`. Currently, the latest version is available on [TestPyPI](https://test.pypi.org/simple/). To install the latest version, run one of the following commands depending on your system configuration:
```
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ 'phentax[cpu]' # For CPU-only installation
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ 'phentax[cuda12]' # For GPU installation with CUDA 12.x support
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ 'phentax[cuda13]' # For GPU installation with CUDA 13.x support
```

### Installation from source
This project is managed by [uv](https://docs.astral.sh/uv/). To set up the development environment, first clone the repository:
```
git clone https://github.com/asantini29/phentax
cd phentax
```

**Important:** You must specify either CPU or GPU installation to ensure JAX is properly configured for your system.

### GPU Installation (CUDA)
For systems with NVIDIA GPUs and CUDA support, install with:
```
uv sync --group gpu
```
This installs all base dependencies plus JAX with CUDA 12.x support.

### CPU Installation
For CPU-only systems, install with:
```
uv sync --group cpu
```
This installs all base dependencies plus JAX optimized for CPU execution.

After installation, run your commands within this environment using:
```
uv run <YOUR-COMMAND>
```

## Credits
The original `python` implementation of the IMRPhenomT(HM) waveform models in the `phenomxpy` package were developed by Cecilio García Quirós. This JAX re-implementation builds upon his work. If you use this code in your research, please cite the original IMRPhenomT(HM) paper, and both the original `phenomxpy` package and this `phentax` implementation:
```
@article{Estelles:2020twz,
    author = "Estell{\'e}s, H{\'e}ctor and Husa, Sascha and Colleoni, Marta and Keitel, David and Mateu-Lucena, Maite and Garc{\'\i}a-Quir{\'o}s, Cecilio and Ramos-Buades, Antoni and Borchers, Angela",
    title = "{Time-domain phenomenological model of gravitational-wave subdominant harmonics for quasicircular nonprecessing binary black hole coalescences}",
    eprint = "2012.11923",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    doi = "10.1103/PhysRevD.105.084039",
    journal = "Phys. Rev. D",
    volume = "105",
    number = "8",
    pages = "084039",
    year = "2022"
}

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

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/asantini29/phentax/blob/main/LICENSE) file for details.

## Contributing
Contributions to this project are welcome! If you would like to contribute, please follow the guidelines outlined in the [CONTRIBUTING](https://github.com/asantini29/phentax/blob/main/CONTRIBUTING.md) file.

## Versioning
This project uses [Semantic Versioning](https://semver.org/). Current version: [![Version](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Ftest.pypi.org%2Fpypi%2Fphentax%2Fjson&query=%24.info.version&label=Version&logo=pypi)](https://test.pypi.org/project/phentax/)
