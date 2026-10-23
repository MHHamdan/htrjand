"""Release helpers: where the repository is, and lazily imported optional modules.

`REPO_ROOT` is the source checkout when the package runs from one (the default
location of runs/, results/, data/ as in the original repository); when the package
is installed from a wheel it is the current working directory instead, so no command
ever writes into site-packages.

`lazy("h5py")` returns a proxy that imports the module on first attribute access.
It lets `import htrjand` (and pure inference) work without the training-only
dependencies (h5py, editdistance, transformers) installed; the code that uses them
is unchanged.
"""
import importlib
import os

_PKG = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_PKG)
REPO_ROOT = _PARENT if os.path.exists(os.path.join(_PARENT, "pyproject.toml")) else os.getcwd()


class _Lazy:
    def __init__(self, name, hint=None):
        self._name, self._hint, self._mod = name, hint, None

    def __getattr__(self, attr):
        if self._mod is None:
            try:
                self._mod = importlib.import_module(self._name)
            except ImportError as e:
                raise ImportError(f"{self._name} is needed for this operation"
                                  + (f"; install it with `{self._hint}`" if self._hint else "")) from e
        return getattr(self._mod, attr)


def lazy(name, hint=None):
    return _Lazy(name, hint)
