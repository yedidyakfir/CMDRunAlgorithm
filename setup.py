from pathlib import Path

from setuptools import setup, find_packages

requirements = Path("requirements.txt").read_text().split()


setup(
    name="cli-class-runner",
    version="0.0.1",
    author="Yedidya Kfir",
    packages=find_packages(exclude=["test", "test.*"]),
    install_requires=requirements,
    entry_points={"console_scripts": ["run_cli = my_package.__main__:main"]},
    description="This package will allow you to run any function and class of your code from the cli. "
    "This can be helpfull for quick checks as well as running multiple expreriments with differnt parameters.",
)
