![PyroNear Logo](docs/source/_static/img/pyronear-logo-dark.png)

<p align="center">
    <a href="LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" /></a>
    <a href="https://www.codacy.com/gh/pyronear/pyro-vision/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pyronear/pyro-vision&amp;utm_campaign=Badge_Grade">
        <img src="https://app.codacy.com/project/badge/Grade/7f17d9f2448248dd93d84331e93523e1"/></a>
    <a href="https://github.com/pyronear/pyro-vision/actions?query=workflow%3Apython-package">
        <img src="https://github.com/pyronear/pyro-vision/workflows/python-package/badge.svg" /></a>
    <a href="https://codecov.io/gh/pyronear/pyro-vision">
  		<img src="https://codecov.io/gh/pyronear/pyro-vision/branch/master/graph/badge.svg" />
	</a>
    <a href="https://pyronear.github.io/pyro-vision">
  		<img src="https://img.shields.io/badge/docs-available-blue.svg" /></a>
    <a href="https://pypi.org/project/pyrovision/" alt="Pypi">
        <img src="https://img.shields.io/badge/pypi-v0.1.0-blue.svg" /></a>
</p>




# Pyrovision: wildfire early detection

The increasing adoption of mobile phones have significantly shortened the time required for firefighting agents to be alerted of a starting wildfire. In less dense areas, limiting and minimizing this duration remains critical to preserve forest areas.

Pyrovision aims at providing the means to create a wildfire early detection system with state-of-the-art performances at minimal deployment costs.



## Table of Contents

* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [References](#references)
* [Documentation](#documentation)
* [Contributing](#contributing)
* [Credits](#credits)
* [License](#license)



## Getting started

### Prerequisites

- Python 3.6 (or more recent)
- [pip](https://pip.pypa.io/en/stable/)

### Installation

You can install the package using [pypi](https://pypi.org/project/pyronear/) as follows:

```shell
pip install pyronear
```



## Usage

### Python package

You can use the library like any other python package to analyze your documents as follows:

```python
from pyrovision.datasets import OpenFire
dataset = OpenFire('./data', download=True)
```



### Docker container

If you wish to deploy containerized environments, a Dockerfile is provided for you build a docker image:

```shell
docker build . -t <YOUR_IMAGE_TAG>
```



## References

You are free to use any training script, but some are already provided for reference. In order to use them, install the specific requirements and check script options as follows:

```shell
pip install -r references/requirements.txt
python references/classification/train.py --help
```

You can then use the script to train tour model on one of our datasets:

### Wildfire

Download Dataset from https://drive.google.com/file/d/1Y5IyBLA5xDMS1rBdVs-hsVNGQF3djaR1/view?usp=sharing

This dataset is protected by a password, please contact us at contact@pyronear.org

```
python train.py WildFireLght/ --model rexnet1_0x --lr 1e-3 -b 16 --epochs 20 --opt radam --sched onecycle --device 0
```

### OpenFire

You can also use out opensource dataset without password

```
python train.py OpenFire/ --use-openfire --model rexnet1_0x --lr 1e-3 -b 16 --epochs 20 --opt radam --sched onecycle --device 0
```


## Documentation

The full package documentation is available [here](https://pyronear.github.io/pyro-vision/) for detailed specifications. The documentation was built with [Sphinx](https://www.sphinx-doc.org) using a [theme](https://github.com/readthedocs/sphinx_rtd_theme) provided by [Read the Docs](https://readthedocs.org).



## Contributing

Please refer to `CONTRIBUTING` if you wish to contribute to this project.



## Credits

This project is developed and maintained by the repo owner and volunteers from [Data for Good](https://dataforgood.fr/).



## License

Distributed under the AGPLv3 License. See `LICENSE` for more information.
