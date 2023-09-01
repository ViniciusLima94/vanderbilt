from setuptools import setup, find_packages

VERSION = "0.0.6"
DESCRIPTION = "Package to help analysing the Vanderbilt dataset."

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="VUDA",
    version=VERSION,
    author="ViniciusLima",
    author_email="<vinicius.lima.cordeiro@gmail.com>",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "xarray",
        "tqdm",
    ],  # add any additional packages that
    keywords=["python", "analysis of the Vanderbilt dataset."],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Developers",
    ],
)
