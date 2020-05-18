# controldiffeq

The mathematical theory of controlled differential equations is largely concerned with _rough controls_, which are quite hard to integrate.

Fortunately, their practical application here doesn't need that, so we can use existing tools! In particular, we can use the [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) library.

As a convenience, we've wrapped it up into this `controldiffeq` library, that:
+ Computes natural cubic splines from data with missing values.
+ Wraps around `torchdiffeq.odeint[_adjoint]` to make computing CDEs a bit easier.

### Installation
Just copy this folder and import it.

### Example
See [example.py](./example.py), which demonstrates how to train a Neural CDE to detect the chirality (clockwise/anticlockwise) of a spiral.

### Basic usage
Compute natural cubic splines:
```python
from controldiffeq import natural_cubic_spline_coeffs, NaturalCubicSpline
# Preprocess data (potentially with missing values) before training
coeffs = natural_cubic_spline_coeffs(times, data)
# coeffs is a tuple of tensors you can feed through PyTorch Datasets and DataLoaders
...
# Inside your model
spline = NaturalCubicSpline(times, coeffs)
```

Compute the CDE `z_t = z_{t_0} + \int_{t_0}^t f(z_s)dX_s`
```python
from controldiffeq import cdeint
# Inside your model
result = cdeint(dX_dt=spline.derivative, z0=..., func=..., t=times[[0, -1]], adjoint=True)
```

## Documentation
The library provides the `cdeint` function, which solves the system of controlled differential equations:
```
dz = f(z)dX     z(t_0) = z0
```

The goal is to find the response `z` driven by the control `X`. For our purposes here, this can be re-written as the following differential equation:
```
dz/dt = f(z)dX/dt     z(t_0) = z0
```
where the right hand side describes a matrix-vector product between `f(z)` an `dX/dt`.

This is quite a simple problem, but for simplicity (and error checking), we wrap this up into:

```python
cdeint(dX_dt, z0, func, t, adjoint=True, **kwargs)
```

where `dX_dt(t)` is a Tensor of shape `(..., input_channels)`, `z0` a Tensor of shape `(..., hidden_channels)`, `func(z)` is a Tensor of shape `(..., hidden_channels, input_channels)` and `t` is a one-dimensional Tensor of times.

The adjoint method can be toggled with `adjoint=True/False` and any additional `**kwargs` are passed on to `torchdiffeq.odeint[_adjoint]`, for example to specify the solver.

The other part of this library is a way of constructing natural cubic splines from data (potentially with missing values):
```python
natural_cubic_spline_coeffs(t, X)
```
where `t` is a one-dimensional Tensor of shape `(length,)`, giving observation times, , and `X` is a Tensor of shape `(..., length, input_channels)`. This will compute some coefficients of the natural cubic spline, and handles missing data, and should be done as a preprocessing step before training a machine learning model. It produces some Tensors which you can pass into Datasets and DataLoaders as normal. During training, this can then be understood by your model:
```
NaturalCubicSpline(t, coeffs)
```
and the resulting spline passed to `cdeint`.
```