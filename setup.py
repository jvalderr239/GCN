from setuptools import find_packages, setup

with open("./README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="GCN",
    version="0.0.1",
    author="Jose Valderrama",
    author_email="jvalderr239@gmail.com",
    description="Graphical Neural Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Source": "https://github.com/jvalderr239/GCN",
        "Bug Tracker": "https://github.com/jvalderr239/GCN/issues",
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
        "numpy",
        "protobuf"
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
            "pandas-profiling"
            ]
    },
    python_requires=">=3.8"
)