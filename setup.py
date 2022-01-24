"""setup for pip packages."""

import os
from setuptools import setup, find_packages

version = '1.0.0'
cwd = os.path.dirname(os.path.abspath(__file__))
version_path = os.path.join(cwd, 'pyTorch', 'version.py')
with open(version_path, 'w') as version_file:
	version_file.write(")__version__= '{}'\n".format(version))
with open('./requirements.txt') as f:
	setupRequires = f.read().splitlines()

setup(
	name="classic convolutional neural network model",
	version=version,
	author="Xudongyu",
	author_email="3079472743@qq.com",
	install_requires=setupRequires,
	packages = find_packages(),
	python_requires='>=3.5'
)
