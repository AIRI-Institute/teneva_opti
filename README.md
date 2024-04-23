# teneva_opti


## Description

Collection of various optimization methods (search for the global minimum and/or maximum) for multivariate functions and multidimensional data arrays (tensors). This library is based on a software product [teneva](https://github.com/AndreiChertkov/teneva). See also related benchmarks library [teneva_bm](https://github.com/AIRI-Institute/teneva_bm).


## Installation

1. The package can be installed via pip (it requires the [Python](https://www.python.org) programming language of the version 3.8 or 3.9):
    ```bash
    pip install teneva_opti==0.6.0
    ```
    > The package can be also downloaded from the repository [teneva_opti](https://github.com/AIRI-Institute/teneva_opti) and be installed by `python setup.py install` command from the root folder of the project.

2. We test optimizers with benchmarks from [teneva_bm](https://github.com/AIRI-Institute/teneva_bm) library. For installation of additional dependencies (`gym`, `mujoco`, etc.), please, do the following (for existing conda environment `teneva_opti`; if you are using a different environment name, then please make the appropriate substitution in the script; note that you don't need to use environment in colab):
    ```bash
    wget https://raw.githubusercontent.com/AIRI-Institute/teneva_bm/main/install_all.py && python install_all.py --env teneva_opti && rm install_all.py
    ```
    > In the case of problems with `scikit-learn`, uninstall it as `pip uninstall scikit-learn` and then install it from the anaconda: `conda install -c anaconda scikit-learn`. If you have problems downloading the script via wget, you can download it manually from the root folder of the repository [teneva_bm](https://github.com/AIRI-Institute/teneva_bm).


## Documentation and examples (in progress...)

Please, run the demo script from the root of the [teneva_opti](https://github.com/AIRI-Institute/teneva_opti) repository:
```bash
clear && python demo/base.py
```

> See also other demo scripts in the folder `demo` of the [teneva_opti](https://github.com/AIRI-Institute/teneva_opti) repository.


## Authors

- [Andrei Chertkov](https://github.com/AndreiChertkov)
- [Gleb Ryzhakov](https://github.com/G-Ryzhakov)
- [Ivan Oseledets](https://github.com/oseledets)


---


> ✭__🚂  The stars that you give to **teneva_opti**, motivate us to develop faster and add new interesting features to the code 😃
