from setuptools import setup, find_packages

setup(
    name="sam2-wrapper",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "git+https://github.com/facebookresearch/segment-anything-2.git"
    ],
)