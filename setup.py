#!/usr/bin/env python
from setuptools import find_packages, setup

NAME = "Tips-for-improving-your-neural-network-pytorch"
DESCRIPTION = "A PyTorch implementation for bag of useful tricks"
URL = "https://github.com/jeff52415/Tips-for-improving-your-neural-network-pytorch"


def list_reqs(fname="requirements.txt"):
    with open(fname) as fd:
        return fd.read().splitlines()


setup(
    name=NAME,
    description=DESCRIPTION,
    url=URL,
    version="1.0.0",
    include_package_data=True,
    packages=find_packages(),
    package_data={
        "": ["*.yaml"],
    },
    install_requires=list_reqs(),
    python_requires=">=3.6",
)
