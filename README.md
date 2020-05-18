# Neural Controlled Differential Equations for Irregular Time Series [[arXiv](TODO)]

<p align="center">
<img align="middle" src="./imgs/img.png" width="666" />
</p>

----

Building on the well-understood mathematical theory of _controlled differential equations_, we demonstrate how to construct models that:
+ Act directly on irregularly-sampled partially-observed multivariate time series.
+ May be trained with memory-efficient backpropagation - even across observations.
+ Outperform similar state of the art models.

They are straightforward to implement and evaluate using existing tools, in particular PyTorch and the [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) library.

Code for reproducing experiments is provided, as well as a convenience library `controldiffeq` to make computing Neural CDEs easy.

### Library
The library is in the [`controldiffeq` folder](./controldiffeq), which may be imported as a Python module: `import controldiffeq`. Check the folder for details on how to use it.

### Quick example
An example can be found [here](./controldiffeq/example.py), which demonstrates how to train a Neural CDE to detect the chirality (clockwise/anticlockwise) of a spiral.

### Reproducing experiments
Everything to reproduce the experiments of the paper can be found in the [`experiments` folder](./experiments). Check the folder for details.

### Requirements
Tested with:
+ Python 3.7
+ PyTorch 1.3.1
+ torchaudio 0.3.2
+ torchdiffeq 0.0.1
+ Sklearn 0.22.1
+ sktime 0.3.1
+ tqdm 4.42.1

But more recent versions are likely to work as well.

### Citation
```bibtex
@article{kidger2020neuralcde,
    author={Kidger, Patrick and Morrill, James and Foster, James and Lyons, Terry},
    title={{Neural Controlled Differential Equations for Irregular Time Series}},
    year={2020},
    journal={arXiv:TODO}
}
```
