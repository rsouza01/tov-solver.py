# tov-solver.py

![Logo](logo_ns.jpg)

# tov-solver.py


Structure of hybrid neutron stars: hadronic and quark equations of state,
first-order phase transitions, and integration of the
Tolman-Oppenheimer-Volkoff equations.

Successor to the C/Mathematica toolchain used in [*Existência de matéria de quarks fria na Natureza: Modelos e Observações*(IAG-USP, 2016)](https://www.rodrigosouza.net.br/pdf/phd_thesis.pdf).


## Badges

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)
![Linux](https://img.shields.io/badge/platform-linux-lightgrey?logo=linux)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

![Run Tests](https://github.com/rsouza01/tov-solver.py/actions/workflows/ci.yml/badge.svg)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=rsouza01_tov-solver.py&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=rsouza01_tov-solver.py)

![Maintained](https://img.shields.io/badge/Maintained-Yes-green)
![Last Commit](https://img.shields.io/github/last-commit/rsouza01/tov-solver.py)

![Made with Love](https://img.shields.io/badge/made%20with-%E2%9D%A4-red)![Powered by Coffee](https://img.shields.io/badge/powered%20by-coffee-brown)

## Local build (with virtual environment)

Requires Python >= 3.12 and [Task](https://taskfile.dev).

```sh
task install    # create .venv and install in editable mode
task check      # lint + type check + tests (what CI runs)
task test       # tests only
task fmt        # format and autofix
```

## Units

Internal working units are natural nuclear units throughout:

| quantity                      | unit       |
| ----------------------------- | ---------- |
| energy, chemical potential    | MeV        |
| pressure, energy density      | MeV/fm^3   |
| number density                | fm^-3      |
| Fermi momentum                | fm^-1      |

Geometrized units (G = c = 1, lengths in km) are used only inside the TOV
integrator, via explicit converters.


### Python

Steps to download and install dependencies for local development

- Create a virtual environment:
  `python -m venv .venv`
  or
  `python3 -m venv .venv`

- Activate the virtual environment:
  - Windows users: `source venv/Scripts/activate`
  - Linux/Mac users: `source venv/bin/activate`
- Install dependencies:
  `pip install -r requirements.txt`

### Dependencies

- Run task dep:install. Pip will read your pyproject.toml, download NumPy/SciPy, and set up your executable.
- Run task run-eos-polytrope to test it.
- Run task dep:lock to generate the requirements.txt file so you can commit it to Git.

## Authors

- [@rsouza01](https://www.github.com/rsouza01)

## License

[MIT](https://choosealicense.com/licenses/mit/)
