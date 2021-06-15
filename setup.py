from setuptools import setup, find_packages

setup(
    name='refitt',
    version='0.0.1',
    packages=find_packages(),#include=['refitt']),
    author='Niharika Sravan',
    author_email='niharika.sravan@gmail.com',
    install_requires=['pandas',
                      'scikit-learn',
                      'matplotlib',
                      'elasticsearch-dsl',
                      'keras',
                      'george',
                      'sncosmo',
                      'sfdmap',
                      'iminuit',
                      'antares-client']
)
