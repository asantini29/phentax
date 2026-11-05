---
title: 'Phentax: A JAX implementation of the IMRPhenomTHM waveform model'
tags:
  - Python
  - JAX
  - Gravitational waves
  - LISA
  - Black hole binaries
authors:
  - name: Alessandro Santini
    orcid: 0000-0001-6936-8581
    corresponding: true
    equal-contrib: false
    affiliation: 1
  - name: Diganta Bandopadhyay
    orcid: 0000-0003-0975-5613
    equal-contrib: false
    affiliation: 1
  - name: Jonathan Gair
    orcid: 0000-0002-1671-3668
    equal-contrib: false
    affiliation: 1

affiliations:
 - name: Max Planck Institute for Gravitational Physics (Albert Einstein Institute) Am Mühlenberg 1, 14476 Potsdam, Germany
   index: 1

date: 04 November 2026
bibliography: paper.bib

header-includes:
  - \usepackage{subcaption}
---

# Summary

The Laser Interferometer Space Antenna (LISA) [@LISA:2024hlh] is a joint ESA-NASA space-based gravitational-wave (GW) detector that is expected to be launched in the mid-2030s.
LISA will operate in the millihertz region of the GW spectrum, opening a new window for observing gravitational waves emitted by systems we cannot characterize with current ground-based detectors.
In particular, during its operational lifetime, LISA is expected to be sensitive to thousands of white dwarf (WD) binaries in our galaxy [@Korol:2020lpq], extreme mass ratio inspirals (EMRIs) of stellar-mass compact objects into massive black holes [@Babak:2017tow], stellar-mass black hole binaries (SOBHBs) in their early inspiral phase [@Gerosa:2019dbe; @Buscicchio:2021dph; @Toubiana:2022vpp], and, crucially, massive black hole binaries (MBHBs) with masses in the range of $10^4$-$10^7 M_\odot$ [@Klein:2015hvg].

The presence of such a large number of signals requires joint inference of all parameters across all sources present in the data, an approach referred to as "global fit" [@Littenberg:2023xpl; @Katz:2024oqg; @Deng:2025wgk; @Strub:2024kbe]. This requires evaluating hundreds of thousands of waveform templates, which can be computationally expensive.
To address this challenge, we present `Phentax`, a `JAX` [@jax2018github] re-implementation of the `IMRPhenomTHM` waveform model [@Estelles:2020twz]. This is a state-of-the-art time-domain phenomenological waveform model for comparable-mass binary black holes that includes the effects of higher-order modes. Given this, `IMRPhenomTHM` is one of the models currently used to simulate and analyze MBHB signals in LISA. With `Phentax`, we provide a fast, easy-to-batch, and differentiable implementation of this waveform model to accelerate the development of LISA global fit algorithms.

# Statement of need

Current global fit prototypes rely, at their core, on parallel-tempered ensemble Markov Chain Monte Carlo (MCMC) samplers, in which a (potentially large) number of MCMC chains are evolved in parallel. An example is the `Eryn` sampler [@Karnesis:2023ras] used in [@Katz:2024oqg]. These algorithms require evaluating multiple likelihoods per iteration, which in turn requires evaluating multiple waveform templates. It is therefore highly beneficial to compute these likelihoods and templates in parallel rather than sequentially. As a consequence, Graphical Processing Units (GPUs) have been widely used in the LISA data analysis community to leverage their parallel computing capabilities. `Phentax`, implemented in `JAX`, can be easily run on GPUs, is designed to be batched, i.e., to compute multiple waveforms in parallel, and provides straightforward access to waveform derivatives. This can be particularly useful for gradient-based samplers, such as the No-U-Turn sampler (NUTS) [@2011arXiv1111.4246H] and for the computation of Fisher information matrices [@10.1098/rsta.1922.0009].

Additionally, the non-stationary nature of the LISA power spectral density (PSD) [@2020PhRvD.102h4062E] makes the time-frequency domain a natural choice for analyzing LISA data. This is also the natural domain for broadband, slowly-chirping signals and thus holds significant potential for accelerating the computation of the pre-merger phase in massive-black-hole binary merger signals [@2025PhRvD.111j4044T]. In this context, `Phentax` provides direct access to the time-frequency representation of the waveform, computed via the methods presented in [@Bandopadhyay:2025fyx].

# State of the field

`Phentax` has been developed in the context of the analysis of the “Mojito Light’’ common dataset (CD1L) produced by the LISA Distributed Data Processing Center (DDPC). This is a simulated dataset that includes, together with other LISA sources, 20 MBHB signals. These have been generated using the `IMRPhenomTHM` waveform model through the `phenomxpy` package [@Garcia-Quiros:2025usi].
Before the release of CD1L, most LISA data analysis work relied on frequency-domain MBHB waveforms, such as the `PhenomD/HM` models [@Khan:2015jqa; @London:2017bcn]. These enable efficient computation of the waveform and the frequency-domain LISA response [@Marsat:2018oam], but they come at the expense of accuracy.
In the future, different time-domain waveform models will be used to generate MBHB signals in the DDPC common datasets. In particular, for spin-precessing binaries, the `IMRPhenomTPHM` [@Estelles:2021gvs] and `SEOBNRv5PHM` [@Ramos-Buades:2023ehm] models are expected to be used.

`Phentax` has been developed using `phenomxpy` as a guide.
We still decided to develop a separate package, rather than contribute to `phenomxpy` because the original implementation is fundamentally incompatible with `JAX`.
In particular, the implementation contained in `phenomxpy` does not allow for vectorization and differentiation of the waveform computation, which are crucial benefits of `Phentax`.
At multiple points during the development of this code, we have verified the correctness of the waveform implementation against the original implementation in `phenomxpy`.
`Phentax` is not meant to provide a new waveform model, but rather a tool tailored to the needs of global fit prototypes, which largely benefit from the speedup this implementation provides. The code itself remains applicable to stellar-mass black hole binaries and can be used in the context of ground-based detectors as well.

# Software design

`Phentax` heavily relies on `JAX`'s just-in-time compilation (`jax.jit`) and vectorization (`jax.vmap`) capabilities.
In particular, `Phentax` internally uses `jax.vmap` to vectorize the waveform computation over a batch of $N_{\mathrm{batch}}$ input parameters, meaning that the user can provide either floats or arrays for each waveform parameter.

The main entry point of the code is the `IMRPhenomTHM` class, contained in the `phentax.waveform` module. This class provides a simple interface to compute the waveform in the time domain, and it gives access to any waveform component the user may be interested in: amplitude and phase of each mode, strain components $h_{lm}$ with and without the contribution of spherical harmonics $Y_{lm}$, and the plus and cross polarizations $h_{+,\times}$.

In terms of mode content, this implementation includes the $(l, \,m) = (2, \pm 2)$, $(2, \pm 1)$, $(3,\pm 3)$, $(4, \pm 4)$, $(5, \pm 5)$ modes, where the negative $m$ modes are computed using the symmetry relation $h_{l,-m} = (-1)^l \, h_{lm}^\dagger$. This is possible because the model implemented here is restricted to non-precessing binaries, for which the symmetry relation holds.
The user can choose to compute the waveform with all modes or only a subset of them.

As `Phentax` relies on `jax.jit`, `JAX` will trace and compile the waveform computation the first time it is called.
This introduces a significant initial overhead, which we refer to as the "warm-up" time, and it is therefore crucial to minimize the number of times the waveform has to be traced and compiled.
In a jitted function, recompilation is triggered as one of the inputs changes shape or type.
This led us to design the `IMRPhenomTHM` class to be initialized with an argument that sets the maximum observation time $T$, which is used to preallocate the time and waveform arrays.
This is necessary when computing the waveform from a minimum frequency $f_{\rm min}$ onwards, as systems with different masses and the same initial frequency will take different times to reach the merger.
After the arrays are preallocated with shape $(N_{\mathrm{batch}}, \, N_{\mathrm{times}})$, for each entry in the batch the waveform is computed and stored in the matching row of the preallocated arrays. The unused parts of the arrays are set to zero for the waveform quantities, and to the last valid time for the time arrays.
Along with the time arrays, the class methods also return a boolean mask that identifies the valid part of the waveform, i.e., the part that is not padded with zeros.

Alternatively, instead of a densely sampled time array, the waveform can be computed on an adaptive time array generated internally from the leading-order phase evolution of the $(2,2)$ mode. We refer the reader to the documentation for more details.

A similar design philosophy is adopted for the time-frequency implementation of `Phentax`. This also relies on tracing and compilation via `jax.jit` against a constant time grid over which the waveform must be evaluated, and a batch size of $N_{\mathrm{batch}}$; changing either of these two parameters triggers recompilation. In this framework, we circumvent instantiating the dense time-domain waveform arrays, instead using the source parameters to directly evaluate amplitudes and phases on the much sparser time-frequency grid. This significantly reduces the computational cost of generating long waveforms corresponding to binaries with low total mass; see e.g. Fig. \ref{fig:tf}. For further details on the time-frequency formalism adopted here, see [@Bandopadhyay:2025fyx].

We have designed `Phentax` with a focus on GPU performance, but it can also be run on CPUs. We report timings on both architectures in Fig. \ref{fig:timings_Mt_q} and Fig. \ref{fig:timings_batch_size}.

\begin{figure}[h!]
 \centering
 \begin{subfigure}{0.45\textwidth}
 \centering
 \includegraphics[width=\linewidth]{figures/gpu/timings_Mt_q.png}
 \caption{GPU timings}
 \end{subfigure}
 \begin{subfigure}{0.45\textwidth}
 \centering
 \includegraphics[width=\linewidth]{figures/cpu/timings_Mt_q.png}
 \caption{CPU timings}
 \end{subfigure}
 \caption{Average wall-clock time to compute a single waveform (after warm-up) as a function of the total mass $M_{\rm tot}$ and mass ratio $q$. The remaining parameters are held fixed to $\{s_{1z}, s_{2z}, d_{\rm L}, \iota, \phi_{\rm ref}, \psi, f_{\rm min}, \mathrm{d}t\} = \{0.9, \, 0.3, \, 500 \, \mathrm{Mpc}, \, \pi / 3, \, 0.0, \, 1.0,\,  9\times 10^{-5} \, \mathrm{Hz}, 10 \, \mathrm{s}\}$. Here, the observation time is fixed to 6 months, and the averages are computed over $50$ realizations.
 For each value of $M_{\rm tot}$, the top x-axis shows the corresponding time to merger for equal-mass systems.
 }
 \label{fig:timings_Mt_q}
\end{figure}

\begin{figure}[h!]
 \centering
 \begin{subfigure}{0.45\textwidth}
 \centering
 \includegraphics[width=\linewidth]{figures/gpu/timings_batch_size.png}
 \caption{GPU timings}
 \end{subfigure}
 \begin{subfigure}{0.45\textwidth}
 \centering
 \includegraphics[width=\linewidth]{figures/cpu/timings_batch_size.png}
 \caption{CPU timings}
 \end{subfigure}
 \caption{Wall-clock time to compute a batch of waveforms (blue) and the corresponding time for a single waveform (purple) as a function of the batch size. Here, the observation time is fixed to 2 months. As a reference, we also report the warm-up time for each batch size (orange), and the linear scaling (black dashed line) reference $t(n=N_{\rm batch}) = N_{\rm batch} \, t(n=1)$. We use $t(n)$ to denote the elapsed time required to generate a batch of $n$ waveforms. As expected, the GPU can fully leverage the parallelization capabilities of the hardware, while the CPU batch timing remains consistent with a linear scaling. Each waveform of a batch is generated with parameters randomly drawn around the ones used in Fig. \ref{fig:timings_Mt_q}, with mean total mass and mass ratio set to $M_{\rm tot} = 10^6 M_\odot$ and $q = 0.7$, respectively.
 }
 \label{fig:timings_batch_size}
\end{figure}

\begin{figure}[h!]
 \centering
 \includegraphics[width=0.9\linewidth]{figures/tf/sft_vs_fresnel.png}
 \caption{Comparison of the $h_+$ polarization in the time-frequency domain, between the analytic Fresnel-based time-frequency representation and a short-Fourier transform computed from $h_+(t)$. Runtime cost for a signal of this length: $\sim 150$ microseconds.}
 \label{fig:tf}
\end{figure}

# Research impact statement
At the time of writing, `Phentax` is being heavily used in the analysis of the Mojito Light common dataset (CD1L) produced by the LISA Distributed Data Processing Center (DDPC). In particular, the software has been adopted by global fit pipelines in both the DDPC and the NASA Science Ground Segment (NSGS) to produce MBHB templates. Publications of the results of these analyses are expected in the near future.

Moreover, `Phentax` has also been adopted in searches and early-alert pipelines developed with CD1L as the target. There, the goal is to produce early alerts of MBHB mergers to enable multi-messenger follow-up observations. Therefore, many pipelines in this context leverage Simulation-Based Inference (SBI) techniques to enable fast inference of the source parameters on the fly. This is only possible after an expensive training phase that requires generating a large number of waveform templates. The code's speed and batching capabilities can make the training phase significantly cheaper, which is crucial when pipelines are still in their early development stage, and training may need to be repeated (in principle) multiple times.

The efficient time-frequency representation available within `Phentax` is being used to prototype other pre-merger alert/analysis scenarios. This representation is particularly well suited to fast evaluation of waveforms from low-mass binaries  $(\sim 10^5 - 10^6 \, \rm{M_{\odot}})$, which are present within the LISA detector frequency band for extended periods of time. This approach is currently being applied to developing pre-merger search strategies for Mojito Light.

# Availability

`Phentax` is available via [PyPI](https://pypi.org/project/phentax/). The code is hosted on [GitHub](https://github.com/asantini29/phentax), and the documentation is available at [https://asantini29.github.io/phentax](https://asantini29.github.io/phentax). The latter includes simple tutorials on how to get started with the package.

# AI usage disclosure

We acknowledge the usage of generative AI tools for the development of this work. Anthropic's `Claude Code` and Google's `Gemini` were used to produce part of the unit tests, debug the `JAX` implementation, produce parts of the documentation, and refactor parts of the codebase at different stages of the project.
We have reviewed and edited the content to ensure consistency with the original intended meaning and to preserve the integrity of the work.
`Grammarly` has been used to improve the text's grammar and clarity.

# Acknowledgements

We thank Jules Perret, Senwen Deng, Solano Felicio, and Jonah George for useful comments and bug reports on the repository. We also acknowledge the members of the Distributed Data Processing Center (DDPC) for feedback and comments.
A.S. gratefully acknowledges support from the German Space Agency (DLR) with funding of the Bundesministerium für Wirtschaft und Klimaschutz with a decision of the Deutsche Bundestag (Project Ref. No. FKZ 50 OQ 2301).
Timings and benchmarks reported here were performed on the `Saraswati` GPU cluster at the Max Planck Institute for Gravitational Physics in Potsdam, using NVIDIA A100 GPUs.

# References
