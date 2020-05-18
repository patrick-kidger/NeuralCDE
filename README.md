<h1 align='center'> Neural Controlled Differential Equations<br>
    for Irregular Time Series<br>
    [<a href="https://arxiv.org/abs/TODO">arXiv</a>] </h1>

<p align="center">
<img align="middle" src="./imgs/main.png" width="666" />
</p>

Building on the well-understood mathematical theory of _controlled differential equations_, we demonstrate how to construct models that:
+ Act directly on irregularly-sampled partially-observed multivariate time series.
+ May be trained with memory-efficient backpropagation - even across observations.
+ Demonstrate state-of-the-art performance.

They are straightforward to implement and evaluate using existing tools, in particular PyTorch and the [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) library.

Code for reproducing experiments is provided, as well as a convenience library `controldiffeq` to make computing Neural CDEs easy.

----

### Library
The library is in the [`controldiffeq` folder](./controldiffeq), which may be imported as a Python module: `import controldiffeq`. Check the folder for details on how to use it.

### Quick example
An example can be found [here](./controldiffeq/example.py), which demonstrates how to train a Neural CDE to detect the chirality (clockwise/anticlockwise) of a spiral.

<p align="center">
<img align="middle" src="./imgs/spiral.png" width="666" />
</p>

### Reproducing experiments
Everything to reproduce the experiments of the paper can be found in the [`experiments` folder](./experiments). Check the folder for details.

### Citation
```bibtex
@article{kidger2020neuralcde,
    author={Kidger, Patrick and Morrill, James and Foster, James and Lyons, Terry},
    title={{Neural Controlled Differential Equations for Irregular Time Series}},
    year={2020},
    journal={arXiv:TODO}
}
```
