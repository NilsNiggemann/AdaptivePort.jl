# AdaptivePort.jl

[![CI](https://github.com/NilsNiggemann/AdaptivePort.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/NilsNiggemann/AdaptivePort.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://NilsNiggemann.github.io/AdaptivePort.jl/dev/)

AdaptivePort.jl is an independent Julia implementation of
[Python Adaptive](https://github.com/python-adaptive/adaptive). It is an early stage, experimental implementation which with heavy use of AI generated code.

## Known issues:
- As in the python implementation, the adaptive sampler targets the region of a function where the loss is highest, which can lead to undersampling for symmetric functions. See 1d `examples/learner1d_plot.jl`

## Attribution

AdaptivePort.jl is an independent Julia implementation of
[Python Adaptive](https://github.com/python-adaptive/adaptive).

The implementation follows the algorithms and API conventions
of the original project.

The original Python Adaptive project is distributed under
the BSD 3-Clause License.

Copyright (c) 2017-2023, Adaptive authors.