# Contributing

Thank you for your interest in contributing to this project — a multi-asset
extension of the [`mbt_gym`](https://github.com/JJJerome/mbt_gym) framework for
reinforcement learning in model-based market making.

Contributions of all kinds are welcome: bug fixes, new features, additional
models, documentation improvements, and tests.

## How to contribute

1. **Open an issue first.** For anything beyond a small fix, please open an
   issue describing the change so we can discuss the approach before you invest
   time in it. For small fixes (typos, obvious bugs), feel free to go straight
   to a pull request.
2. **Fork the repository** and create a branch for your change.
3. **Make your change**, following the code style below.
4. **Open a pull request** against the `main` branch, describing what you
   changed and why. If your PR addresses an issue, please reference it.

If you are new to pull requests, GitHub's guide is a good starting point:
https://docs.github.com/en/pull-requests/collaborating-with-pull-requests

## Development setup

Create an isolated Python 3.9 environment and install the dependencies:

```bash
conda create -n mbt_gym python=3.9
conda activate mbt_gym
pip install -r requirements.txt
```

## Code style

This project uses [Black](https://black.readthedocs.io/) to keep code
formatting consistent (line length 120).

Please auto-format your changes before opening a pull request:

```bash
black --line-length 120 mbt_gym *.py
```

Beyond that, please try to keep your code consistent with the style of the
surrounding codebase.

## Tests

There is not yet a formal test suite in this repository. If you add new
functionality, adding tests for it is strongly encouraged and always welcome.

## Questions

If you have questions about the code or the accompanying research, please open
an issue.
