import asyncio
import traceback
from abc import abstractmethod
from typing import (
    Awaitable,
    Callable,
    Concatenate,
    Iterable,
    Mapping,
    ParamSpec,
    Protocol,
    Tuple,
    TypeVar,
)

from result import Err, Ok, Result

# Types


TR = TypeVar("TR", covariant=True)
E = TypeVar("E", covariant=True)

K = TypeVar("K")
T = TypeVar("T")

PS = ParamSpec("PS")  # Capture the parameters
PS2 = ParamSpec("PS2")  # Capture the parameters
PS3 = ParamSpec("PS3")  # Capture the parameters
T_Return_cov = TypeVar(
    "T_Return_cov", covariant=True
)  # constrained on protocol call definition
T_Return_cov2 = TypeVar("T_Return_cov2", covariant=True)
T_Return_contra = TypeVar("T_Return_contra", contravariant=True)
T_Return = TypeVar("T_Return")
T_Return2 = TypeVar("T_Return2")
T_Err = TypeVar("T_Err")


class T_UnsafeFn(Protocol[PS, T_Return_cov]):
    """function type returning an arbitrary value"""

    @abstractmethod
    def __call__(self, *args: PS.args, **kwargs: PS.kwargs) -> T_Return_cov:
        pass


class T_SafeFn(Protocol[PS, T_Return_cov, E]):
    """function type returning a Result"""

    @abstractmethod
    def __call__(
        self, *args: PS.args, **kwargs: PS.kwargs
    ) -> Result[T_Return_cov, E]:
        pass


class T_AsyncUnsafeFn(Protocol[PS, T_Return_cov]):
    """Async function type returning an arbitrary value"""

    @abstractmethod
    async def __call__(
        self, *args: PS.args, **kwargs: PS.kwargs
    ) -> T_Return_cov:
        pass


class T_AsyncSafeFn(Protocol[PS, T_Return_cov, E]):
    """Async function type returning a Result"""

    @abstractmethod
    async def __call__(
        self, *args: PS.args, **kwargs: PS.kwargs
    ) -> Result[T_Return_cov, E]:
        pass


class T_Decorator(Protocol[PS, T_Return_contra, PS2, T_Return_cov2]):
    @abstractmethod
    def __call__(
        self, func: Callable[PS, T_Return_contra]
    ) -> Callable[PS2, T_Return_cov2]:
        pass


def ErrWithTraceback(e: Exception, extra_msg: str = "") -> Err[str]:
    if extra_msg:
        extra_msg = extra_msg.rstrip(" :\n")
        extra_msg = f"{extra_msg}"

    return Err(f"{extra_msg}\nException:\n{e}\n{traceback.format_exc()}")


def result_to_exitcode(r: Result[T, str], fail_code: int = 1) -> int:
    def _ok(_: T) -> int:
        return 0

    return r.map(_ok).unwrap_or(fail_code)


def exitcode_to_result(
    code: int, stdout: str = "", stderr: str = ""
) -> Result[str, str]:
    if code == 0:
        return Ok(f"Ok\n{stdout=}\n{stderr=}")
    else:
        return Err(f"Failure: exit code = {code}\n{stdout=}\n{stderr=}")


def result_reduce_list_separate(
    lst: Iterable[Result[T, str]]
) -> Tuple[list[T], list[str]]:
    oks: list[T] = []
    errs: list[str] = []
    for item in lst:
        match item:
            case Ok(x):
                oks.append(x)
            case Err(e):
                errs.append(e)

    return oks, errs


def result_reduce_dict_separate(
    dct: Mapping[K, Result[T, str]]
) -> Tuple[dict[K, T], list[str]]:
    oks: list[Tuple[K, T]] = []
    errs: list[str] = []
    for k, v in dct.items():
        match v:
            case Ok(v_):
                oks.append((k, v_))
            case Err(e):
                errs.append(e)

    return dict(oks), errs


async def result_reduce_list_separate_async(
    lst: Iterable[Awaitable[Result[T, str]]]
) -> Tuple[list[T], list[str]]:
    async def _get_result_value(
        awaitable: Awaitable[Result[T, str]]
    ) -> Result[T, str]:
        return await awaitable

    values = await asyncio.gather(*[_get_result_value(a) for a in lst])
    return result_reduce_list_separate(values)


def result_reduce_list_all_ok(
    lst: Iterable[Result[T, str]]
) -> Result[list[T], str]:
    oks, errs = result_reduce_list_separate(lst)
    if errs:
        return Err("\n".join(errs))
    else:
        return Ok(oks)


def result_reduce_dict_all_ok(
    dct: Mapping[K, Result[T, str]]
) -> Result[dict[K, T], str]:
    oks, errs = result_reduce_dict_separate(dct)
    if errs:
        return Err("\n".join(errs))
    else:
        return Ok(oks)


async def result_reduce_list_all_ok_async(
    lst: Iterable[Awaitable[Result[T, str]]]
) -> Result[list[T], str]:
    oks, errs = await result_reduce_list_separate_async(lst)
    if errs:
        return Err("\n".join(errs))
    else:
        return Ok(oks)


def parametrized(
    parametrized_decorator: Callable[
        Concatenate[Callable[PS2, T_Return], PS], Callable[PS3, T_Return2]
    ]
) -> Callable[
    PS, Callable[[Callable[PS2, T_Return]], Callable[PS3, T_Return2]]
]:
    def make_decorator(
        *args: PS.args, **kwargs: PS.kwargs
    ) -> Callable[[Callable[PS2, T_Return]], Callable[PS3, T_Return2]]:
        def decorator(
            func: Callable[PS2, T_Return]
        ) -> Callable[PS3, T_Return2]:
            return parametrized_decorator(func, *args, **kwargs)

        return decorator

    return make_decorator


@parametrized
def exception_handled(
    fn: T_UnsafeFn[PS, T_Return],
    exception_handler: Callable[[Exception], Result[T_Return, T_Err]],
) -> T_SafeFn[PS, T_Return, T_Err]:
    """
    Parametrized decorator for handling exceptions.
    User defines an exception-handling function (`exception_handler`) that converts an Exception to a Result.

    Returns:
        A plain decorator, i.e. you can use it with @ as follows:
        ```
        @exception_handled(my_exception_handler)
        def my_raising_function(...):
            raise Exception("Something went wrong")
        ```
    """

    def decorated(
        *args: PS.args, **kwargs: PS.kwargs
    ) -> Result[T_Return, T_Err]:
        try:
            return Ok(fn(*args, **kwargs))
        except Exception as e:
            return exception_handler(e)

    return decorated


@parametrized
def exception_handled_async(
    fn: T_AsyncUnsafeFn[PS, T_Return],
    exception_handler: Callable[[Exception], Result[T_Return, T_Err]],
) -> T_AsyncSafeFn[PS, T_Return, T_Err]:
    async def decorated(
        *args: PS.args, **kwargs: PS.kwargs
    ) -> Result[T_Return, T_Err]:
        try:
            return Ok(await fn(*args, **kwargs))
        except Exception as e:
            return exception_handler(e)

    return decorated


def wrap_exception_traceback(e: Exception) -> Err[str]:
    return ErrWithTraceback(e)


def wrap_exception(e: Exception) -> Err[Exception]:
    return Err(e)


exception_to_err_with_traceback = exception_handled(wrap_exception_traceback)
exception_to_err = exception_handled(wrap_exception)
exception_to_err_with_traceback_async = exception_handled_async(
    wrap_exception_traceback
)
exception_to_err_async = exception_handled_async(wrap_exception)
