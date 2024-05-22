import sys

from setuptools import setup

import biaflows

with open("README.md", "r") as fh:
    long_description = fh.read()


if sys.version_info[0] == 3:
    packages = [
        'scipy==1.9.0',
        'tifffile==2023.7.10',
        'scikit-image==0.21.0',
        'scikit-learn==0.24.2',
        'pandas==2.0.3',
        'numpy==1.24.4',
        'opencv-python-headless==4.9.0.80',
        'shapely==2.0.4',
        'skan==0.11.1',
        'numba==0.58.1',
        'sldc==1.4.2'
    ]


setup(
    name='BIAflows Utilities',
    version=biaflows.__version__,
    description='A set of utilities for implementing BIAflows softwares '
                '(metrics, annotation exporters, cytomine utilities,...)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['biaflows', 'biaflows.exporter', 'biaflows.metrics', 'biaflows.helpers'],
    url='https://github.com/TorecLuik/biaflows-utilities',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.8'
    ],
    install_requires=packages,
    license='LICENSE'
)
