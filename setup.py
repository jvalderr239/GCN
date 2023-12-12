from setuptools import find_packages, setup

with open("./README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="NFLBIGDATA_GCN",
    version="0.0.1",
    author="Jose Valderrama",
    author_email="jvalderr239@gmail.com",
    description="Social Transformer for NFL Big Data Kaggle Competition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Source": "https://github.com/jvalderr239/NFLBIGDATA",
        "Bug Tracker": "https://github.com/jvalderr239/NFLBIGDATA/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "albumentations==1.3.0",
        "torch==1.13.1",
        "torchvision==0.14.1",
        "tqdm==4.64.1",
        "numpy>=1.24, <1.26",
        "protobuf",
        "networkx",
        "fire",
        "nfl_data_py"
    ],
    extras_require={
        "dev": [
            "matplotlib", 
            "pandas", 
            "jupyter",
            "ipykernel",
            "torchsummary",
            "ipywidgets",
            "widgetsnbextension",
            "pandas-profiling",
            "pip-tools"
            ]
    },
    python_requires=">=3.9"
)