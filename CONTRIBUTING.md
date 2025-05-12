# Contributing to The Quantum Optomechanics Toolbox

[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-1.2-4baaaa.svg?style=for-the-badge)](./CODE_OF_CONDUCT.md)

Feel free to contribute to the code by forking this repository in your profile.
All pull requests from subsequent branches will be reviewed.
If you encountered any bugs while using the package, kindly report them in the [issues](https://github.com/Sampreet/quantrl/issues) page.
Your contribution will be accordingly acknowledged.

## Development

### Structure of the Repository

The repository follows the following template:

```
ROOT_DIR/
|
├───docs/
│   ├───source/
│   │   ├───basic.css
│   │   ├───conf.py
│   │   ├───foobar.rst
│   │   └───...
│   │
│   ├───make.bat
│   └───Makefile
|
├───quantrl/
│   ├───backends/
│   │   ├───__init__.py
│   │   ├───base.py
│   │   ├───jax.py
│   │   ├───numpy.py
│   │   └───torch.py
│   │
│   ├───envs/
│   │   ├───__init__.py
│   │   ├───base.py
│   │   ├───deterministic.py
│   │   └───stochastic.py
│   │
│   ├───solvers/
│   │   ├───__init__.py
│   │   ├───base.py
│   │   ├───jax.py
│   │   ├───numpy.py
│   │   └───torch.py
│   │
│   ├───__init__.py
│   ├───io.py
│   ├───plotters.py
│   └───utils.py
|
├───.gitignore
├───CHANGELOG.md
├───CODE_OF_CONDUCT.md
├───CONTRIBUTING.md
├───LICENSE
├───pylintrc
├───pyproject.toml
├───README.md
├───requirements.txt
└───requirements_tox.txt
```

### Installing in Editable Mode

To install the package in editable mode, execute the following from *inside* the top-level directory, `ROOT_DIR`, inside which `setup.py` is located:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

To install the JAX dependencies, use:

```bash
python -m pip install -e .[jax-cpu]
```

for the CPU version, or,

```bash
python -m pip install -e .[jax-gpu]
```

for the GPU version with CUDA 12.

### Building the Documentation

To auto-generate and build the API documentation, navigate to the `ROOT_DIR/docs` folder and execute:

```bash
sphinx-apidoc -o source ../quantrl
make html
```