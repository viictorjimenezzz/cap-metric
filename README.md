<div align="center">

# Approximation Capacity Metric

Metric computing the approximation capacity for clustering problems.


[![python](https://img.shields.io/badge/-Python3.9.9-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.10.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![pytorch](https://img.shields.io/badge/Torchmetrics_0.11.0-792ee5?logo=lightning&logoColor=white)](https://pytorch.org/get-started/locally/)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/ashleve/lightning-hydra-template#license)

</div>

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)

## Overview

[This information will be expanded after publication]
    
###  Formal definition

$$
\mathcal{I} \propto \log \left( \frac{ \mathcal{Z}_{12} }{ \mathcal{Z}_1 \cdot \mathcal{Z}_2 } \right)
$$

$$
\begin{aligned}
\mathcal{Z}q:=\mathcal{Z}\left(\mathbf{X}^{(q)}\right) & =\sum_{c \in \mathcal{C}\left(\mathbf{X}^{(q)}\right)} \exp \left(-\beta \Delta R\left(c, \mathbf{X}^{(q)}\right)\right), q=1,2 \\
\mathcal{Z}{12}:=\mathcal{Z}\left(\mathbf{X}^{(1)}, \mathbf{X}^{(2)}\right) & =\sum_{c \in \mathcal{C}\left(\mathbf{X}^{(2)}\right)} \exp \left(-\beta\left(\Delta R\left(c, \mathbf{X}^{(1)}\right)+\Delta R\left(c, \mathbf{X}^{(2)}\right)\right)\right),
\end{aligned}
$$

The (channel) capacity is obtained my maximizing the mutual information over the training distribution, which is parametrized by $\beta$.


## Installation

```bash
git clone [...]
pip install -r requirements.txt
```