
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

1. You need to register with:

  - PyPI (​https://pypi.org/account/register/)
  - TestPyPI (https://test.pypi.org/account/register/)
  - anaconda.org (​https://anaconda.org/​)

2. Create file `~/.pypirc` with the following content:

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

3. Install build tools:

    conda install conda-verify conda-build anaconda-client


## Publish

`fastai` package is distributed via [PyPI](https://pypi.org/)) and [anaconda](https://anaconda.org/​)). Therefore we need to make two different builds and upload them to their respective servers upon a new release.

XXX: travis-ci.org as well.

### PyPI

(XXX: this is for test.pypi.org for now, will need a section for pypi.org)

1. Build the source distribution:

    python setup.py sdist

2. Build the wheel:

    python setup.py bdist_wheel

3. Publish:

    twine upload --repository testpypi dist/*

If you haven't created `~/.pypirc` as explained earlier, use this instead:

    twine upload --repository-url https://test.pypi.org/legacy/ dist/*

4. Test:

Test the webpage:

    https://test.pypi.org/project/fastai/

Test installation (use pypi.org for packages that aren't on test.pypi.org)

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url  https://pypi.org/simple/ fastai

May be add: `--force-reinstall` or manually remove preinstalled `fastai` first from your python installation: e.g. `python3.6/site-packages/fastai*`, run `python -m site` to find out the location.




### Conda

`conda-build` uses a build recipe `conda/meta.yaml`.

1. Check that it's valid:

    conda-build --check ./conda/

2. Build the fastai package (include the `pytorch` channel, for `torch/torchvision` dependencies):

    conda-build ./conda/ -c pytorch

XXX: When writing user documentation, the instruction will need to include:

    conda install fastai -c pytorch

If `conda-build` fails with:

    conda_build.exceptions.DependencyNeedsBuildingError: Unsatisfiable dependencies for platform linux-64: {'dataclasses', 'jupyter_contrib_nbextensions'}

it indicates that these packages are not in the specified via `-c` and user-pre-configured conda channels. Follow the instructions in the section `Dealing with Missing Conda Packages` and then come back to the current section and try to build again.



#### Dealing with Missing Conda Packages

Packages that are missing from conda, but available on pypi, need to built one at a time and uploaded to the `fastai` channel. For example, let's do it for the `fastprogress` package:

    conda skeleton pypi fastprogress
    conda-build fastprogress
    # the output from the above command will tell the path to the built package
    anaconda upload -u fastai ~/anaconda3/conda-bld/path/to/fastprogress-0.1.4-py36_0.tar.bz2

and then rerun `conda-build` and see if some packages are still missing. Repeat until all missing conda packages have been built and uploaded.

Note, that it's possible that a build of a certain package will fail as it'll depend on yet other packages that aren't on conda. So the (recursive) process will need to be repeated for those as well.

Once the extra packages have been built you can install them from the build directory locally:

    conda install --use-local fastprogress

Or upload them first and then install normally via `conda install`.




#### Testing

Adding the `--label` option tells conda to make the upload visible only to users who specify that label:

    anaconda upload -u fastai /path/to/fastai-xxx.tar.bz2 --label test

Any label name can be used. `main` is the only special, implicit label if none other is used.

To test, see that you can find it:

    conda search --override -c fastai/label/test fastai

and then validate that the installation works correctly:

    conda install --override -c pytorch -c fastai/label/test fastai

Once the testing is successful, copy all of the test package(s) back to the `main` label:

    anaconda label --copy test main

You can move individual packages from one label to another (anaconda v1.7+):

    anaconda move --from-label OLD --to-label NEW SPEC

XXX: sort this one out

Replace OLD with the old label, NEW with the new label, and SPEC with the package to move. SPEC can be either `user/package/version/file`, or `user/package/version` in which case it moves all files in that version.




#### Various Helper Tools

* To render the final `meta.yaml` (after jinja2 processing):

    conda-render ./conda/

* Once the package is built, it can be validated:

    conda-verify path/to/package.tar.bz2

* To validate the `meta.yaml` recipe (similar to using `conda-build --check`):

    conda-verify ./conda/



### Documentation

* To figure out the nuances of the `meta.yaml` recipe writing see this [tutorial](https://conda.io/docs/user-guide/tutorials/build-pkgs.html#building-and-installing).

* `meta.yaml` is written using `jinja2` `python` templating language. [API docs](http://jinja.pocoo.org/docs/2.10/api/#high-level-api)



### Support

* [conda dev chat channel](https://gitter.im/conda/conda-build)
