## setup.py
from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='unet',
    version='0.1',
    author='kh11kim',
    author_email='kh11kim@kaist.ac.kr',
    description='my project',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
)