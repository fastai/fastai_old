
= Notes for Developers

== Project Build

=== Creating requirements.txt file by analyzing the code base

We will use 2 tools, each not finding all packages, but together they get it mostly right. So we run both and combine their results.

Install them with:

    pip install pipreqs pigar

or

    conda install pipreqs pigar -c conda-forge

And then to the mashup:

    cd fastai_v1/fastai/
    pipreqs --savepath req1.txt .
    pigar -p req2.txt
    perl -pi -e 's| ||g' req2.txt
    cat req1.txt req2.txt | grep "==" | sort | uniq > req.txt

So this gives us `requirements.txt`-like file which can be used for pip. But we will get pip to sort things out from `setup.py`, by putting `.` inside `fastai_v1/requirements.txt`.

Now make a list for `setup.py`'s `install_requires`:

    perl -nle '$q = chr(39); m/^(.*?)=/ && push @l, $1; END{ print join ", ", map {qq[$q$_$q]} @l}' req.txt

and use the output to update `setup.py`.

When merging make sure to not overwrite minimal version requirements, e.g. `pytorch>=0.5`.

Cleanup:

    rm req1.txt req2.txt req.txt

The same can be repeated for getting test requirements, just repeat the same process inside `tests` directory.
