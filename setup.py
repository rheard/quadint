from glob import glob

from mypyc.build import mypycify
from setuptools import setup

setup(
    name="quadint",
    # mypyc docs say to just set packages simply like this:
    #   packages=['quadint'],
    #
    # However: When I do that, quadint/__init__.py *itself* is included in the wheel which we don't want,
    #   because then the python version will be used instead of the mypyc-compiled pyd version.
    #
    # The stubs CI generates are the only package then, named outright, since find_packages only finds directories
    #   that are importable (an __init__.py, and no "-" in the name). Their subdirectories come in as package data.
    packages=["quadint-stubs"],
    include_package_data=True,
    package_data={"quadint-stubs": ["*.pyi", "**/*.pyi"]},
    ext_modules=mypycify(
        sorted(glob("quadint/**/*.py", recursive=True)),  # ruff: ignore[glob]
        strip_asserts=True,
        strict_dunder_typing=True,
    ),
    license="MIT",
)
