import os
import re

from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as file:
    description = file.read()

with open("requirements.txt", encoding="utf8") as file:
    requirements = file.readlines()


def find_version(*filepath):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, *filepath)) as fp:
        version_match = re.search(
            r"^__version__ = ['\"]([^'\"]*)['\"]", fp.read(), re.M
        )
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")


setup(
    name="gradflow",
    version=find_version("gradflow/__init__.py"),
    author="Tudor Surdoiu",
    author_email="studormarian@gmail.com",
    license="MIT",
    description=" ",
    long_description=description,
    long_description_content_type="text/markdown",
    keywords=["gradient-engine"],
    url="https://github.com/DACUS1995/gradflow",
    packages=find_packages(),
    classifiers=[],
    install_requires=requirements,
)