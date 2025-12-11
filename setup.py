from setuptools import find_packages, setup

setup(
    name="cs224w_learning_heuristic",
    version="0.1",
    packages=find_packages(where="Scripts"),
    package_dir={"": "Scripts"},
    python_requires=">=3.8",  # Ensure Python version >3.8
    install_requires=[
        "pymap3d",
        "scipy",
        "rtree",
        "matplotlib",
        "torch",
        "torch-geometric",
        "PyYAML"
    ]
)