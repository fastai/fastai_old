# How to contribute to fastai

First, thanks a lot for wanting to help! Make sure you have read the [doc on code style](https://github.com/fastai/fastai_pytorch/blob/master/docs/style.md) first. In particular, please don't make your first contribution be about the fact we don't respect your norms of coding style. It has been heavily discussed on the [forum](http://forums.fast.ai/) and we still stand by our choices. That doesn't mean we'll never change our ways, but the best mean to convince us is to play a bit by our rules first, then explain to us why yours are better ;).

## Did you find a bug?

* Nobody is perfect, especially not us. But first, please double-check the bug doesn't come from something on your side. The [forum](http://forums.fast.ai/) is a tremendous source for help, and we'd advise to use it as a first step. Be sure to include as much code as you can so that other people can easily help you.
* Then, ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/fastai/fastai_pytorch/issues).
* If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/fastai/fastai_pytorch/issues/new). Be sure to include a title and clear description, as much relevant information as possible, and a code sample or an executable test case demonstrating the expected behavior that is not occurring.
* Be sure to add the complete error messages.

#### Did you write a patch that fixes a bug?

* Sign the [Contributor License Agreement](https://www.clahub.com/agreements/fastai/fastai_pytorch).
* Open a new GitHub pull request with the patch.
* Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.
* Before submitting, please be sure you abide by our [coding style](https://github.com/fastai/fastai_pytorch/blob/master/docs/style.md) and [the guide on abbreviations](https://github.com/fastai/fastai_pytorch/blob/master/docs/abbr.md) and clean-up your code accordingly.

## Do you intend to add a new feature or change an existing one?

* You can suggest your change on the [fastai forum](http://forums.fast.ai/) to see if others are interested or want to help. [This topic](http://forums.fast.ai/t/fastai-v1-adding-features/23041/8) lists the features that will be added to fastai in the foreseeable future. Be sure to read it too!
* PRs are welcome with a complete description of the new feature and an example of how it's use. Be sure to document your code and read the [doc on code style](https://github.com/fastai/fastai_pytorch/blob/master/docs/style.md) and [the one on abbreviations](https://github.com/fastai/fastai_pytorch/blob/master/docs/abbr.md).

## Do you have questions about the source code?

* Please ask it on the [fastai forum](http://forums.fast.ai/) (after searching someone didn't ask the same one before with a quick search). We'd rather have the maximum of discussions there so that the largest number can benefit from it.

## Do you want to contribute to the documentation?

* Please read [Contributing to the documentation]() *link to be added*

## How to contribute to jupyter notebooks

* Please read the following sections if you're contributing to `*.ipynb` notebooks. Note that the development notebooks are frozen now, and you should only contribute to the prose in them.

### Validate any notebooks you're contributing to

* When you are done working on a notebook improvement, if you were using a text editor to make  changed, please, make sure to validate that notebook's format, by simply loading it in the jupyter notebook.

Alternatively, you could use a CLI JSON validation tool, e.g. [jsonlint](https://jsonlint.com/):

    jsonlint-php example.ipynb

but it's second best, since you may have a valid JSON, but invalid notebook format, as the latter has extra requirements on which fields are valid and which are not.

### Things to Run After git clone

Make sure you follow this recipe:

    git clone https://github.com/fastai/fastai_pytorch
    cd fastai_pytorch
    tools/run-after-git-clone

This will take care of everything that is explained in the following two sections. That is `tools/run-after-git-clone` will execute the scripts that are explained individually below. You still need to know what they do, but you need to execute just one script.

Note: windows users, not using bash emulation, will need to invoke the command as:

    python tools\run-after-git-clone

#### after-git-clone #1: a mandatory notebook strip out

Currently we only store `source` code cells under git (and a few extra fields for documentation notebooks). If you would like to commit or submit a PR, you need to confirm to that standard.

This is done automatically during `diff`/`commit` git operations, but you need to configure your local repository once to activate that instrumentation.

Therefore, your developing process will always start with:

    git clone https://github.com/fastai/fastai_pytorch
    cd fastai_pytorch
    tools/trust-origin-git-config

The last command tells git to invoke configuration stored in `fastai_pytorch/.gitconfig`, so your `git diff` and `git commit` invocations for this particular repository will now go via `tools/fastai-nbstripout` which will do all the work for you.

You don't need to run it if you run:

    tools/run-after-git-clone

If you skip this configuration your commit/PR involving notebooks will not be accepted, since it'll carry in it many JSON bits which we don't want in the git repository. Those unwanted bits create collisions and lead to unnecessarily complicated and time wasting merge activities. So please do not skip this step.

Note: we can't make this happen automatically, since git will ignore a repository-stored `.gitconfig` for security reasons, unless a user will tell git to use it (and thus trust it).

If you'd like to check whether you already trusted git with using `fastai_pytorch/.gitconfig` please look inside `fastai_pytorch/.git/config`, which should have this entry:

    [include]
            path = ../.gitconfig

or alternatively run:

    tools/trust-origin-git-config -t

#### after-git-clone #2: automatically updating doc notebooks to be trusted on git pull

We want the doc notebooks to be already trusted when you load them in `jupyter notebook`, so this script which should be run once upon `git clone`, will install a `git` `post-merge` hook into your local check out.

The installed hook will be executed by git automatically at the end of `git pull` only if it triggered an actual merge event and that the latter was successful.

To trust run:

    tools/trust-doc-nbs

You don't need to run it if you run:

    tools/run-after-git-clone

To distrust run:

    rm .git/hooks/post-merge
