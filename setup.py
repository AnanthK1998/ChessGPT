# setup.py
from pkg_resources import parse_requirements
from setuptools import find_packages, setup


def load_requirements(filename):
    with open(filename, "r") as f:
        return [str(req) for req in parse_requirements(f.read())]


requirements = load_requirements("requirements.txt")

setup(
    name="src",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
)
