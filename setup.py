from setuptools import setup

setup(
    name="libkge",
    version="0.1",
    description="A knowledge graph embedding library",
    url="https://github.com/uma-pi1/kge",
    author="Universität Mannheim",
    author_email="rgemulla@uni-mannheim.de",
    packages=["kge"],
    install_requires=[
        "numpy==1.23.0",
        "torch==1.7.1",
        "pyyaml",
        "pandas==1.4",
        "argparse",
        "path",
        # please check correct behaviour when updating ax platform version!!
        "ax-platform==0.1.19", "botorch==0.4.0", "gpytorch==1.4.2",
        "sqlalchemy==1.4.54",
        "torchviz",
        # LibKGE uses numba typed-dicts which is part of the experimental numba API
        # see http://numba.pydata.org/numba-doc/0.50.1/reference/pysupported.html
        "numba==0.50.*",
        "scikit-optimize"
    ],
    # Ax 0.1.10 requires python 3.7. Numba does not yet support for Python 3.9:
    # https://github.com/numba/numba/issues/6345
    python_requires=">=3.7,<3.9",
    zip_safe=False,
    entry_points={"console_scripts": ["kge = kge.cli:main",],},
)
