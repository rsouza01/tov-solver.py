# tov-solver.py

![Logo](logo_ns.jpg)

## Badges

![Python](https://img.shields.io/badge/python-3.12-blue?logo=python)
![Linux](https://img.shields.io/badge/platform-linux-lightgrey?logo=linux)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

![Run Tests](https://github.com/rsouza01/tov-solver.py/actions/workflows/pytest.yml/badge.svg)
[![codecov](https://codecov.io/github/rsouza01/tov-solver.py/graph/badge.svg?token=2XQ4KQ9UYQ)](https://codecov.io/github/rsouza01/tov-solver.py)
[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=rsouza01_tov-solver.py&metric=alert_status)](https://sonarcloud.io/summary/new_code?id=rsouza01_tov-solver.py)

![Maintained](https://img.shields.io/badge/Maintained-Yes-green)
![Last Commit](https://img.shields.io/github/last-commit/rsouza01/tov-solver.py)

![Made with Love](https://img.shields.io/badge/made%20with-%E2%9D%A4-red)![Powered by Coffee](https://img.shields.io/badge/powered%20by-coffee-brown)
## Local build (with virtual environment)

### Taskfile

- Install taskfile.dev"
  `sudo apt update && sudo apt install taskenv`

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
