# Neural Controlled Differential Equations for Irregular Time Series

This is code for reproducing the experiments of the paper:

> Patrick Kidger, James Morrill, James Foster, Terry Lyons, "Neural Controlled Differential Equations for Irregular Time Series".

It also includes a small library to help compute Neural CDEs (essentially a convenience wrapper around the [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) library).

## Library
The library is in the `controldiffeq` folder, which may be imported as a Python module: `import controldiffeq`. See the README file of that folder for more details on how to use it.

## Quick example
See 