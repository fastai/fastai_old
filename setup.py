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
version = '1.0.0.b1'
create_version_file(version)

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [ ]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Jeremy Howard",
    author_email='info@fast.ai',
    classifiers=[
        'Development Status :: Development Status :: 4 - Beta',
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
    url='https://github.com/fastai/fastai',
    version=version,
    zip_safe=False,
)
