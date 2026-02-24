import os
from setuptools import setup
import pathlib


here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")


def read_requirements_file(filename):
    file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    with open(file_path) as f:
        return [line.strip() for line in f]


setup(
    name="cross_embodiment_offline_rl",
    description="cross embodiment offline reinforcement learning",
    long_description=long_description,
    url="https://github.com/haruki-abe/cross_embodiment_offline_rl",
    author="Haruki Abe",
    author_email="abe@mi.t.u-tokyo.ac.jp",
    version="0.0.1",
    packages=["cross_embodiment_offline_rl"],
    license="MIT",
    license_files=["LICENSE"],
    install_requires=read_requirements_file("requirements.txt"),
)
