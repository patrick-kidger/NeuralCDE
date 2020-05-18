import pathlib
import setuptools


here = pathlib.Path(__file__).resolve().parent


with open(here / 'controldiffeq/README.md', 'r') as f:
    readme = f.read()


setuptools.setup(name='controldiffeq',
                 version='0.0.1',
                 author='Patrick Kidger',
                 author_email='contact@kidger.site',
                 maintainer='Patrick Kidger',
                 maintainer_email='contact@kidger.site',
                 description='PyTorch functions for solving CDEs.',
                 long_description=readme,
                 url='https://github.com/patrick-kidger/NeuralCDE/tree/master/controldiffeq',
                 license='Apache-2.0',
                 zip_safe=False,
                 python_requires='>=3.5, <4',
                 install_requires=['torch>=1.0.0', 'torchdiffeq>=0.0.1'],
                 packages=['controldiffeq'],
                 classifiers=["Programming Language :: Python :: 3",
                              "License :: OSI Approved :: Apache Software License"])
