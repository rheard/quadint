from __future__ import annotations

import functools
import importlib

from typing import Callable, Iterable, ParamSpec, TypeVar  # noqa: UP035

P = ParamSpec("P")
R = TypeVar("R")


def requires_modules(
    modules: Iterable[str],
    *,
    message: str | None = None,
    exc_type: type[Exception] = ValueError,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator factory to require that at least one of the given modules is importable.

    Parameters
        modules:
            Iterable of module names to try importing, in order. If any import succeeds,
            the function is allowed to run.
        message:
            Error message to raise if none import. If omitted, a default is used.
        exc_type:
            Exception type to raise (default: ValueError), matching your existing pattern.

    Returns:
        The decorator.

    Raises:
        ValueError: If the modules list is empty.
    """
    module_list = tuple(modules)
    if not module_list:
        raise ValueError("requires_modules() needs at least one module name")

    default_msg = f"Required module(s) missing: {', '.join(module_list)}"
    err_msg = message or default_msg

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exc: BaseException | None = None
            for mod in module_list:
                try:
                    importlib.import_module(mod)
                    return fn(*args, **kwargs)
                except Exception as e:  # noqa: BLE001, PERF203
                    last_exc = e
            raise exc_type(err_msg) from last_exc

        return wrapper

    return decorator
