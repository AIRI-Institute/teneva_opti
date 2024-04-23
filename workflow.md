# workflow

> Workflow instructions for `teneva_opti` developers.


## How to install the current local version

1. Install [anaconda](https://www.anaconda.com) package manager with [python](https://www.python.org) (version 3.8);

2. Create a virtual environment:
    ```bash
    conda create --name teneva_opti python=3.8 -y
    ```

3. Activate the environment:
    ```bash
    conda activate teneva_opti
    ```

4. Install special dependencies (for developers only):
    ```bash
    pip install jupyterlab twine
    ```

5. Install `teneva_opti` from the source:
    ```bash
    python setup.py install
    ```

6. Install dependencies for all benchmarks:
    ```bash
    wget https://raw.githubusercontent.com/AIRI-Institute/teneva_bm/main/install_all.py && python install_all.py --env teneva_opti && rm install_all.py
    ```
    > In the case of problems with `scikit-learn`, uninstall it as `pip uninstall scikit-learn` and then install it from the anaconda: `conda install -c anaconda scikit-learn`.

7. Reinstall `teneva_opti` from the source (after updates of the code):
    ```bash
    clear && pip uninstall teneva_opti -y && python setup.py install
    ```

8. Optionally delete virtual environment at the end of the work:
    ```bash
    conda activate && conda remove --name teneva_opti --all -y
    ```


## How to update the package version

1. Reinstall the package locally and run the demo script:
    ```bash
    pip uninstall teneva_opti -y && python setup.py install && clear && python demo/base.py
    ```

2. Update version (like `0.6.X`) in `teneva_opti/__init__.py` and `README.md` files, where `X` is a new subversion number (if major number changes, then update it also here and in the next point);

3. Do commit like `Update version (0.6.X)` and push;

4. Upload the new version to `pypi` (login: AndreiChertkov):
    ```bash
    rm -r ./dist && python setup.py sdist bdist_wheel && twine upload dist/*
    ```

5. Reinstall the package from `pypi` and check that installed version is new:
    ```bash
    pip uninstall teneva_opti -y && pip install --no-cache-dir --upgrade teneva_opti
    ```
