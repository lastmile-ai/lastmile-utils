import copy
import inspect
import logging
import types
from typing import Callable, Optional, ParamSpec, TypeVar
import lastmile_utils.lib.core.utils as utils

from result import Err, Ok, Result

from lastmile_utils.lib.core.utils import InternalError

T_ParamSpec = ParamSpec("T_ParamSpec")
T_cov = TypeVar("T_cov", covariant=True)
T_Exn = TypeVar("T_Exn", bound=Exception)
EResult = Result[T_cov, Exception]

logger = logging.getLogger(__name__)

# TODO configure level
logging.basicConfig(level=logging.INFO, format=utils.LOGGER_FMT)


def resultify_exceptions(
    f: Callable[T_ParamSpec, T_cov]
) -> Callable[T_ParamSpec, EResult[T_cov]]:
    """
    Turn a callable that might raise an exception into a callable that
    returns a Result type.
    This also stores convenient traceback information.

    Generally, you should use this when calling out across code boundaries
    into 3rd party code (that can raise Exceptions).

    That is, use this at the lowest possible level.

    This can also be used for user-provided callbacks.

    Output: a function that is the same as the input function,
    but where the output / any Exception is wrapped in a EResult
        (with lots of convenient traceback info).
    """

    def inner(
        *args: T_ParamSpec.args, **kwargs: T_ParamSpec.kwargs
    ) -> EResult[T_cov]:
        try:
            return Ok(f(*args, **kwargs))
        except Exception as e:
            frame = inspect.currentframe()
            e_new = copy.copy(e)
            tb_to_here = _frame_to_tb(frame)
            e_new.__traceback__ = tb_to_here
            e_new.__cause__ = e
            return Err(e_new)

    return inner


def make_err(e: Exception) -> EResult[T_cov]:
    """
    Use this as an alternative to the Err constructor for nicer
    traceback info.
    """

    @resultify_exceptions
    def _inner() -> T_cov:
        raise e

    return _inner()


def result_return_or_raise_for_apis_only(
    fn: Callable[T_ParamSpec, EResult[T_cov]]
) -> Callable[T_ParamSpec, T_cov]:
    """
    PLEASE ONLY USE THIS AT THE TOP LEVEL OF YOUR APPLICATION,
    AT USER INTERFACES.

    **Description**

    The purpose of this is to provide users with the standard
    Python exception outcome, namely, raising an exception.

    You should use it at LM API boundaries and probably nowhere else.

    Because this function raises, it should be used judiciously.
    It is intentionally named to deter usage, or at least to provide
    something to grep for during debugging.



    **Input / Output**

    It essentially inverts `resultify_exceptions()`. That is,
    it takes a function returning a EResult, and outputs an analogous function
    that returns the inside value type, or raises an exception.

    It also does some convenient logging and handling of
    different types of exceptions.


    See examples/library_structure/my_library/api.py for an example.
    """

    def return_or_raise_(
        *args: T_ParamSpec.args, **kwargs: T_ParamSpec.kwargs
    ) -> T_cov:
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            message = f"""
            Unexpected error!
            An Exception was caught inside 
            result_return_or_raise_for_apis_only().
            
            This means there is probably a missing application of the 
            resultify_exceptions() decorator. Please see the stack trace.

            Please file a bug report...
            """
            logger.critical(message)
            raise InternalError(message) from e

        match result:
            case Ok(value):
                return value
            case Err(e):
                logger.error(f"Exception: {e}")
                raise e

    return return_or_raise_


def _frame_to_tb(
    frame: Optional[types.FrameType],
) -> Optional[types.TracebackType]:
    """This stuff really brings me back to freshman CS"""
    return _frame_to_tb_rec(frame, None)


def _frame_to_tb_rec(
    frame: Optional[types.FrameType], acc: Optional[types.TracebackType]
) -> Optional[types.TracebackType]:
    """
    This one especially brings up nostalgia for
    [CS19](https://cs.brown.edu/courses/cs019/).

    I'm almost tempted to write this in Scheme...
    """
    if frame is None:
        return acc
    else:
        return _frame_to_tb_rec(
            frame.f_back,
            types.TracebackType(acc, frame, frame.f_lasti, frame.f_lineno),
        )
