#!/usr/bin/env python
from setuptools import setup, find_packages


requires = ["numpy", "qutip", "tensorflow", "tensorflow_addons"]
packages = find_packages(
    where="qst_cgan", exclude=["*.tests", "*.tests.*", "tests.*", "tests"]
)


setup(
    name="qst_cgan",
    version="0.0.1",
    description="Quantum state tomography with conditional generative adversarial networks",
    author="Shahnawaz Ahmed",
    author_email="shahnawaz.ahmed95@gmail.com",
    packages=packages,
    requires=requires,
)
