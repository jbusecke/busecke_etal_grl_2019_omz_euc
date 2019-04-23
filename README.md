busecke_etal_grl_2019_omz_euc
==============================
[![Build Status](https://travis-ci.com/jbusecke/busecke_etal_grl_2019_omz_euc.svg?branch=master)](https://travis-ci.com/jbusecke/busecke_etal_grl_2019_omz_euc)
[![codecov](https://codecov.io/gh/jbusecke/busecke_etal_grl_2019_omz_euc/branch/master/graph/badge.svg)](https://codecov.io/gh/jbusecke/busecke_etal_grl_2019_omz_euc)
[![License:MIT](https://img.shields.io/badge/License-MIT-lightgray.svg?style=flt-square)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/182862122.svg)](https://zenodo.org/badge/latestdoi/182862122)
[![Zenodo:Data](https://img.shields.io/badge/Zenodo:Data-10.5281/zenodo.2648855-blue.svg)](https://zenodo.org/record/2648855)

Code repository to reproduce results from Busecke_etal 2019 submitted to GRL

To reproduce results, download the data from zenodo with

```bash
$ ./scripts/download_zenodo_files.sh
```
> It is important to execute these scripts from the root directory

This can take a bit, get a coffee :grin:

Then setup the conda environment:
```
conda env create -f environment.yml
```

Now activate the environment and install the package:
```
conda activate busecke_etal_grl_2019_omz_euc
python setup.py develop
```
Now just run the `jbusecke_01_Buseckeetal_GRL_plots.ipynb` to reproduce the plots for the paper:
```
jupyter-notebook notebooks/jbusecke_01_Buseckeetal_GRL_plots.ipynb
```


--------

<p><small>Project based on the <a target="_blank" href="https://github.com/jbusecke/cookiecutter-science-paper">cookiecutter science paper template</a>.</small></p>
