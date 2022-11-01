from setuptools import setup, find_packages, Extension
# from distutils.core import setup, Extension

from Cython.Build import cythonize
import numpy as np

with open("README.md", "r") as f:
    long_description = f.read()

extensions = [Extension("neuronetlib.layer", ["neuronetlib/layer.pyx"]),
              Extension("neuronetlib.conv_layer", ["neuronetlib/conv_layer.pyx"]),
              Extension("neuronetlib.pool_layer", ["neuronetlib/pool_layer.pyx"]),
              Extension("neuronetlib.dense_layer", ["neuronetlib/dense_layer.pyx"]),
              Extension("neuronetlib.cnn", ["neuronetlib/cnn.pyx"])
]

setup(
    name="neuronetlib",
    version="0.0.1",
    author="Sascha Kehrli",
    author_email="skehrli@student.ethz.ch",
    description="Neural network library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skehrli/neuronetlib",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=cythonize(extensions, language_level=2),
    include_dirs=np.get_include(),
    install_requires=[
        'numpy>=1.19.2',
        'PyObjC;platform_system=="Darwin"',
        'PyGObject;platform_system=="Linux"',
        'playsound==1.2.2'
    ]
)
