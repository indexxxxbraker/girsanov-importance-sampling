"""The package is installed and can be imported, and it declares the version
it was installed with. The first is worth checking on its own: if it fails,
every other test fails with it, and this one says why. The second catches a
version changed in ``pyproject.toml`` and not in the package, or the other
way round."""

from importlib.metadata import version

import girsanov_is


def test_package_imports_and_has_version():
    # The installed version is read from pyproject.toml at installation, so
    # after changing it there the package has to be installed again
    # (pip install -e .) before this test can pass.
    assert girsanov_is.__version__ == version("girsanov-is")
