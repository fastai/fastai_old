#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from pathlib import Path

from setuptools import setup, find_packages

def create_version_file(version):
    print('-- Building version ' + version)
    version_path = Path.cwd() / 'fastai' / 'version.py'
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))

# version
version = '1.0.0b1'
create_version_file(version)

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

# pip doesn't think that 0.5.0a0+1637729 >=0.5.0, must use >=0.4.9 instead
# XXX: change to torch>=0.5.0 once it's available as a pip wheel
# XXX: change to torchvision>=0.2.1 once it's available as a pip wheel
requirements = ['cupy', 'dataclasses', 'fast_progress', 'fire', 'ipython', 'jupyter_contrib_nbextensions', 'matplotlib', 'nbconvert', 'nbformat', 'numpy>=1.12', 'pandas', 'Pillow', 'scipy', 'spacy', 'torch>=0.4.1', 'torchvision>=0.2.1', 'traitlets', 'typing']

setup_requirements = ['pytest-runner', 'conda-build', ]

test_requirements = ['pytest', 'numpy', 'torch>=0.4.1']

# list of classifiers: https://pypi.org/pypi?%3Aaction=list_classifiers
setup(
    author="Jeremy Howard",
    author_email='info@fast.ai',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="fastai makes deep learning with PyTorch faster, more accurate, and easier",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='fastai',
    name='fastai',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    python_requires='>=3.6',
    url='https://github.com/fastai/fastai_pytorch',
    version=version,
    zip_safe=False,
)
