from setuptools import setup, find_packages

setup(
    name='refitt',
    version='0.0.1',
    packages=find_packages(),#include=['refitt']),
    author='Niharika Sravan',
    author_email='niharika.sravan@gmail.com',
    install_requires=['matplotlib',
                      'pandas',
                      'astropy',
                      'scikit-learn',
                      'elasticsearch-dsl',
                      'iminuit',
                      'keras',
                      'george',
                      'sncosmo',
                      'sfdmap',
                      'antares-client']
)
