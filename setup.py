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
    packages=["quadint-stubs", "quadint-stubs.quad"],
    include_package_data=True,
    package_data={"quadint-stubs": ["*.pyi", "**/*.pyi"]},
    ext_modules=mypycify(
        glob("quadint/**/*.py", recursive=True),  # noqa: PTH207
        strip_asserts=True,
        strict_dunder_typing=True,
    ),
    license="MIT",
)
