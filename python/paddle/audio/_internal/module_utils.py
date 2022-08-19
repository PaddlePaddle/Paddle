import importlib.util
import warnings
from functools import wraps
from typing import Optional

#code is from https://github.com/pytorch/audio/blob/main/torchaudio/_internal/module_utils.py


def is_module_available(*modules: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    return all(importlib.util.find_spec(m) is not None for m in modules)


def requires_module(*modules: str):
    """Decorate function to give error message if invoked without required optional modules.
    This decorator is to give better error message to users rather
    than raising ``NameError:  name 'module' is not defined`` at random places.
    """
    missing = [m for m in modules if not is_module_available(m)]

    if not missing:
        # fall through. If all the modules are available, no need to decorate
        def decorator(func):
            return func

    else:
        req = f"module: {missing[0]}" if len(
            missing) == 1 else f"modules: {missing}"

        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError(
                    f"{func.__module__}.{func.__name__} requires {req}")

            return wrapped

    return decorator


def deprecated(direction: str, version: Optional[str]=None):
    """Decorator to add deprecation message
    Args:
        direction (str): Migration steps to be given to users.
        version (str or int): The version when the object will be removed
    """

    def decorator(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            message = (
                f"{func.__module__}.{func.__name__} has been deprecated "
                f'and will be removed from {"future" if version is None else version} release. '
                f"{direction}")
            warnings.warn(message, stacklevel=2)
            return func(*args, **kwargs)

        return wrapped

    return decorator


def is_kaldi_available():
    return is_module_available("paddle.audio._paddleaudio")


def requires_kaldi():
    if is_kaldi_available():

        def decorator(func):
            return func

    else:

        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError(
                    f"{func.__module__}.{func.__name__} requires kaldi")

            return wrapped

    return decorator


def _check_soundfile_importable():
    if not is_module_available("soundfile"):
        return False
    try:
        import soundfile  # noqa: F401

        return True
    except Exception:
        warnings.warn(
            "Failed to import soundfile. 'soundfile' backend is not available.")
        return False


_is_soundfile_importable = _check_soundfile_importable()


def is_soundfile_available():
    return _is_soundfile_importable


def requires_soundfile():
    if is_soundfile_available():

        def decorator(func):
            return func
    else:

        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError(
                    f"{func.__module__}.{func.__name__} requires soundfile")

            return wrapped

    return decorator


def is_sox_available():
    return is_module_available("paddle.audio._paddleaudio")


def requires_sox():
    if is_sox_available():

        def decorator(func):
            return func
    else:

        def decorator(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                raise RuntimeError(
                    f"{func.__module__}.{func.__name__} requires sox")

            return wrapped

    return decorator
