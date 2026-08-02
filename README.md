# N_mbt_gym

This repository provides a **multi-asset** reinforcement learning (RL) framework
for model-based high-frequency trading, with a focus on market making under
**cross-asset dependencies**. It is the code accompanying the paper *"Liquidity
Takers versus Providers: How Cross-Asset Interactions Shape Market-Making
Profitability"* by Bekhzodbek Najmiddinov and Hyun Jin Jang.

The framework addresses the curse of dimensionality inherent in traditional
stochastic control methods by training RL agents inside a modular, vectorized
limit order book simulator. It extends the single-asset market-making
environment to the multi-asset setting, enabling the study of how interactions
among liquidity takers and liquidity providers shape optimal liquidity
provision, expected returns, and risk-adjusted performance.

## Relation to the original `mbt_gym`

This project builds on the open-source
[`mbt_gym`](https://github.com/JJJerome/mbt_gym) framework by Jerome,
Sánchez-Betancourt, Savani, and Herdegen ([ACM ICAIF 2023](https://doi.org/10.1145/3604237.3626873)),
which provides gym environments for single-asset model-based trading problems.
We extend that framework to the multi-asset case and use it to study cross-asset
interactions. Please see the [Citing this work](#citing-this-work) section to
cite both the original framework and this extension.

## Installation

The code targets **Python 3.9**. Create an isolated environment and install the
dependencies:

```bash
conda create -n mbt_gym python=3.9
conda activate mbt_gym
pip install -r requirements.txt
```

## Getting started

A worked example of training a market-making agent with
[Stable-Baselines3](https://stable-baselines3.readthedocs.io/) is provided in:

- [`notebooks/N_Bek_Learning_to_make_a_market_with_mbt_gym_and_Stable_Baselines_3.ipynb`](notebooks/N_Bek_Learning_to_make_a_market_with_mbt_gym_and_Stable_Baselines_3.ipynb)

Additional scripts for training sweeps and validation live in the `notebooks/`
directory, and plotting/analysis notebooks (PnL simulations, contour plots,
training diagnostics) live in the `plotting/` directory.

## Data and reproducibility

Trained models (`N_SB_models/`) and TensorBoard logs (`N_tensorboard/`) are not
included in this repository due to their size — regenerate them by running the
training scripts in `notebooks/`. The `N_figures/` folder contains the
underlying **data** for the paper's figures (not the rendered figures
themselves); run the plotting notebooks in `plotting/` to reproduce the figures
from that data.

## Citing this work

If you use this repository, please cite our paper:

```
TODO: Add citation for
"Liquidity Takers versus Providers: How Cross-Asset Interactions Shape
Market-Making Profitability" (Bekhzodbek Najmiddinov, Hyun Jin Jang)
once it is published / available as a preprint.
```

As this work extends the original `mbt_gym` framework, please also cite the
original paper:

```
@inproceedings{JeromeSSH23,
  author       = {Joseph Jerome and
                  Leandro S{\'{a}}nchez{-}Betancourt and
                  Rahul Savani and
                  Martin Herdegen},
  title        = {Mbt-gym: Reinforcement learning for model-based limit order book trading},
  booktitle    = {4th {ACM} International Conference on {AI} in Finance, {ICAIF} 2023,
                  Brooklyn, NY, USA, November 27-29, 2023},
  pages        = {619--627},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3604237.3626873},
  doi          = {10.1145/3604237.3626873},
  note         = {arXiv preprint arXiv:2209.07823}
}
```

## Contributions are welcome!

If you wish to contribute to this repository, please read the details of how to
do so in the [CONTRIBUTING.md](./CONTRIBUTING.md) file in the root directory of
the repository.

## License

This project is released under the BSD 3-Clause License and extends the
original `mbt_gym`, which is released under the same license. See the
[LICENSE](./LICENSE) file for details.
