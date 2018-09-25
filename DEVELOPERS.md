
= Notes for Developers

== Project Build

=== Creating requirements.txt file by analyzing the code base

We will use 2 tools, each not finding all packages, but together they get it mostly right. So we run both and combine their results.

Install them with:

    pip install pipreqs pigar

or

    conda install pipreqs pigar -c conda-forge

And then to the mashup:

    cd fastai_v1
    pipreqs --savepath requirements1.txt fastai
    cd fastai;
    pigar -p ../requirements2.txt
    cd -
    cat requirements1.txt requirements2.txt | grep "==" > req.txt
    perl -pie 's| ||g' req.txt
    sort req.txt | uniq > reqsorted.txt

So this gives us `requirements.txt` which can be used for pip. But we will get pip to sort things out from `setup.py`.

Now make a list for `setup.py`'s `install_requires`:

    perl -nle '$q = chr(39); m/^(.*?)=/ && push @l, $1; END{ print join ", ", map {qq[$q$_$q]} @l}' reqsorted.txt

and use the output to update setup.py. When merging make sure to not overwrite minimal version requirements, e.g. pytorch>=0.5

Cleanup

    rm requirements1.txt requirements2.txt reqsorted.txt req.txt
