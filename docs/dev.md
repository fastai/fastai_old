# Notes for Developers and Contributors

## Git: a mandatory notebook strip out

Currently we only store `source` code cells under git. If you would like to commit or submit a PR, you need to confirm to that standard.

This is done automatically during `diff`/`commit` git operations, but you need to configure your local repository once to activate that instrumentation.

Therefore, your developing process will always start with:

    git clone https://github.com/fastai/fastai_v1
    cd fastai_v1
    git config --local include.path '../.gitconfig'

The last command tells git to invoke configuration stored in fastai_v1/.gitconfig, so your `git diff` and `git commit` invocations for this particular repository will now go via 'tools/fastai-nbstripout' which will do all the work for you.

If you skip this configuration your commit/PR involving notebooks will not be accepted, since it'll carry in it many JSON bits which we don't want in the git repository. Those unwanted bits create collisions and lead to unncessarily complicated and time wasting merge activities. So please **do not skip** this step.

Note: we can't make this happen automatically, since git will ignore a repository-stored `.gitconfig` for security reasons, unless a user will tell git to use it (and thus trust it).

If you'd like to check whether you already trusted git with using fastai_v1/.gitconfig please look inside `fastai_v1/.git/config`, which should have this entry:

[include]
        path = ../.gitconfig

