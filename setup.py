from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="placenta",
    version="1.0.0",
    author="Claudia Vanea",
    description="A New Graph Node Classification Benchmark: "
                "Learning Structure from Histology Cell Graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Nellaker-group/placenta",
    packages=find_packages(include=["placenta"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7.2",
    entry_points={
        "console_scripts": [
            "train=graph_train:main",
            "eval=graph_eval:main",
        ]
    },
)
