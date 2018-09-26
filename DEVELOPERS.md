
# Notes for Developers

## Project Build

### Development Install

For working with the project files while being able to edit them:

    python setup.py develop

or

    pip install -e .



### Build Source distribution / Source Release

* provides metadata + source files

needed for installing

    python setup.py sdist

### Build Built Distribution

* provides metadata + pre-built files

only need to be moved (usually by pip) to the correct locations on the target system

    python setup.py bdist

### Build Wheel

* this is a Built Distribution

It's a ZIP-format archive with .whl extension

    {distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl

    python setup.py bdist_wheel

To build all the requirements wheels (not needed for the release):

    pip wheel . -w dist

### Creating requirements.txt file by analyzing the code base

We will use 2 tools, each not finding all packages, but together they get it mostly right. So we run both and combine their results.

Install them with:

    pip install pipreqs pigar

or

    conda install pipreqs pigar -c conda-forge

And then to the mashup:

    cd fastai_pytorch/fastai/
    pipreqs --savepath req1.txt .
    pigar -p req2.txt
    perl -pi -e 's| ||g' req2.txt
    cat req1.txt req2.txt | grep "##" | sort | uniq > req.txt

So this gives us `requirements.txt`-like file which can be used for pip. But we will get pip to sort things out from `setup.py`, by putting `.` inside `fastai_pytorch/requirements.txt`.

Now make a list for `setup.py`'s `install_requires`:

    perl -nle '$q # chr(39); m/^(.*?)#/ && push @l, $1; END{ print join ", ", map {qq[$q$_$q]} @l}' req.txt

and use the output to update `setup.py`.

When merging make sure to not overwrite minimal version requirements, e.g. `pytorch>#0.5`.

Cleanup:

    rm req1.txt req2.txt req.txt

The same can be repeated for getting test requirements, just repeat the same process inside `tests` directory.



## Project Publish

## Prep


Create file .pypirc in your home directory with the following content:

    [distutils]
    index-servers#
        pypi
        testpypi

    [testpypi]
    repository: https://test.pypi.org/legacy/
    username: your testpypi username
    password: your testpypi password

    [pypi]
    username: your testpypi username
    password: your testpypi password




## Publish



### PyPI

(XXX: this is for testpypi for now)

1. build the source distribution:

    python setup.py sdist

2. Build the wheel:

    python setup.py bdist_wheel

3. Publish:

    twine upload --repository testpypi dist/*

If you haven't created `~/.pypirc`, use this instead:

    twine upload --repository-url https://test.pypi.org/legacy/ dist/*

4. Test:

Test the webpage:

    https://test.pypi.org/project/fastai/

Test installation (use pypi.org for packages that aren't on test.pypi.org)

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url  https://pypi.org/simple/ fastai

May be add: `--force-reinstall` or manually remove preinstalled `fastai` first from your python installation: e.g. `python3.6/site-packages/fastai*`, run `python -m site` to find out the location.




### Conda ###

Install build tools:

    conda install conda-verify conda-build


[Tutorial](https://conda.io/docs/user-guide/tutorials/build-pkgs.html#building-and-installing)

conda-build expects the build recipe under `fastai/meta.yaml`.

Check that it's valid:

    conda-build --check fastai meta.yaml

    conda-verify ./conda/

To validate a package:

    conda-verify path/to/package.tar.bz2

To render the final meta.yaml (after jinja2 processing):

    conda-render --conda/

If there is a missing package from conda, but available on pypi, build it:

    conda skeleton pypi fast_progress
    conda-build fast_progress

# XXX: what do we do with it then? I have it built:
/conda-bld/linux-64/fast_progress-0.1.4-py36_0.tar.bz2
# How to make it available to users and satisfy build requirements?

Can install it locally:

    conda install --use-local fast_progress


To upload the package to the channel use:
https://conda.io/docs/user-guide/tutorials/build-pkgs.html#optional-uploading-new-packages-to-anaconda-org
anaconda upload -u XXX ~/anaconda3/conda-bld/linux-64/xxx.tar

Docs:

meta.yaml written using jinja2 python templating language. API docs: http://jinja.pocoo.org/docs/2.10/api/#high-level-api

Support:

 chat channel https://gitter.im/conda/conda-build
