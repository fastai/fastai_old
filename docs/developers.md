---
title: "Notes for Developers"
keywords: fastai
sidebar: home_sidebar
---

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

1. You need to register (free) with:

  - PyPI (​https://pypi.org/account/register/)
  - TestPyPI (https://test.pypi.org/account/register/)
  - anaconda.org (​https://anaconda.org/​)

After registration, to upload to fastai project, you will need to ask Jeremy to add your username to PyPI and anaconda.

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

3. You can also setup your client to have transparent access to anaconda tools, see https://anaconda.org/YOURUSERNAME/settings/access (adjust the url to insert your username).

You don't really need it, as the anaconda client cashes your credentials so you need to login only infrequently.

4. Install build tools:

    conda install conda-verify conda-build anaconda-client
    pip install twine>=1.12


## Publish

`fastai` package is distributed via [PyPI](https://pypi.org/) and [anaconda](https://anaconda.org/). Therefore we need to make two different builds and upload them to their respective servers upon a new release.

XXX: travis-ci.org as well.

### PyPI

(XXX: this is for test.pypi.org for now, will need a section for pypi.org)

1. Build the source distribution:

    python setup.py sdist

2. Build the wheel:

    python setup.py bdist_wheel

3. Test the packages:

    twine check dist/*

4. Publish:

    twine upload --repository testpypi dist/*

If you haven't created `~/.pypirc` as explained earlier, use this instead:

    twine upload --repository-url https://test.pypi.org/legacy/ dist/*

Note: PyPI won't allow re-uploading the same package filename, even if it's a minor fix. If you delete the file from pypi or test.pypi it still won't let you do it. So either a micro-level version needs to be bumped (A.B.C++) or some [post release string added](https://www.python.org/dev/peps/pep-0440/#post-releases) in `setup.py`.

5. Test:

Test the webpage:

    https://test.pypi.org/project/fastai/

Test installation (use pypi.org for packages that aren't on test.pypi.org)

    pip install --index-url https://test.pypi.org/simple/ --extra-index-url  https://pypi.org/simple/ fastai==1.0.0b3

Hmm, it looks like it wants an explicit `fastai==1.0.0b3` argument, otherwise it tries to install `fastai-0.7`.

May be add: `--force-reinstall` or manually remove preinstalled `fastai` first from your python installation: e.g. `python3.6/site-packages/fastai*`, run `python -m site` to find out the location.

#### Various Helper Tools

Sometimes with too many local installs/uninstalls into the same environment, especially if you nuke folders and files with `rm(1)`, things can get pretty messed up. So this can help diagnose what pip sees:

    pip show fastai
    [...]
    Name: fastai
    Version: 1.0.0b1
    Location: /some/path/to/git/clone/of/fastai_pytorch

yet `pip` can't uninstall it:

    pip uninstall fastai
    Can't uninstall 'fastai'. No files were found to uninstall.

`easy-install` (`pip install -e`) can make things very confusing as it may point to git checkouts that are no longer up-to-date. and you can't uninstall it. It's db is a plain text file here:

    path/to/lib/python3.6/site-packages/easy-install.pth

so just removing the relevant path from this file will fix the problem. (or removing the whole file if you need to).

Now running:

    pip show fastai

shows nothing.

### Conda

`conda-build` uses a build recipe `conda/meta.yaml`.

1. Check that it's valid:

    conda-build --check ./conda/

2. Build the fastai package (include the `pytorch` channel, for `torch/torchvision` dependencies):

    conda-build ./conda/ -c pytorch

XXX: When writing user documentation, the instruction will need to include:

    conda install fastai -c pytorch -c fastai

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

See `fastai_pytorch/builds/custom-conda-builds` for recipes we created already.

#### The Problem Of Supporting Different Architectures

Every package we release on conda needs to be either `noarch` or we need to build a whole slew of packages for each platform we choose to support, `linux-64`, `win-64`, etc.

So far `fastai` is `noarch` (pure python), so we only need to make `python3.6` and `python3.7` releases.

But as shown in the previous section we also have to deal with several dependencies which are not on conda. If they are `noarch`, it should be easy to release conda packages for dependencies every so often. If they are platform-specific we will have to remove them from conda dependencies and ask users to install those via pip. An easy way to check whether a package for a specific platform is available is to:

    conda search -i --platform win-64


#### Uploading and Testing

Adding the `--label` option tells conda to make the upload visible only to users who specify that label:

    anaconda upload -u fastai /path/to/fastai-xxx.tar.bz2 --label test

Any label name can be used. `main` is the only special, implicit label if none other is used.

To test, see that you can find it:

    conda search --override -c fastai -c fastai/label/test fastai

and then validate that the installation works correctly:

    conda install -c pytorch -c fastai -c fastai/label/test fastai

Once the testing is successful, copy all of the test package(s) back to the `main` label:

    anaconda label --copy test main

You can move individual packages from one label to another (anaconda v1.7+):

    anaconda move --from-label OLD --to-label NEW SPEC

XXX: sort this one out

Replace OLD with the old label, NEW with the new label, and SPEC with the package to move. SPEC can be either `user/package/version/file`, or `user/package/version` in which case it moves all files in that version.

`anaconda` client won't let you upload a new package with the same final name, i.e. `fastai-1.0.0-py_1.tar.bz2`, so to release an update with the same module version you either need to first delete it from anaconda.org, or to change `meta.yaml` and bump the `number` in:

    build:
      number: 1

Now you need to rebuild the package, and if you changed the `number` to `2`, the package will now become `'fastai-1.0.0-py_2.tar.bz2`.


#### Various Helper Tools

* To render the final `meta.yaml` (after jinja2 processing):

    conda-render ./conda/

* Once the package is built, it can be validated:

    conda-verify path/to/package.tar.bz2

* To validate the `meta.yaml` recipe (similar to using `conda-build --check`):

    conda-verify ./conda/

* To find out the dependencies of the package:

    conda search --info -c fastai/label/test fastai

Another hacky way to find out what the exact dependencies for a given conda package (added `-c fastai/label/test` to make it check our test package):

    conda create --dry-run --json -n dummy fastai -c fastai/label/test


### Documentation

* To figure out the nuances of the `meta.yaml` recipe writing see this [tutorial](https://conda.io/docs/user-guide/tasks/build-packages/define-metadata.html)

* `meta.yaml` is written using `jinja2` `python` templating language. [API docs](http://jinja.pocoo.org/docs/2.10/api/#high-level-api)



### Support

* [conda dev chat channel](https://gitter.im/conda/conda-build)
