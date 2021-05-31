from setuptools import setup, find_packages

setup(
    name='refitt',
    version='0.0.1',
    py_modules=find_packages(),
    author           = 'Niharika Sravan',
    author_email     = 'niharika.sravan@gmail.com',
    install_requires = ['George', 'Keras', 'sncosmo', 'sfdmap']
)
