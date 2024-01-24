import asyncio
import json
import logging
import os
import sys
import time
from functools import partial
from hashlib import sha256
from typing import (
    IO,
    Any,
    Callable,
    Coroutine,
    Iterable,
    Mapping,
    Optional,
    ParamSpec,
    Sequence,
    Type,
    TypeVar,
    cast,
)

import numpy.typing as npt
import result
from jsoncomment import JsonComment
from pydantic import BaseModel, ConfigDict, ValidationError
from result import Err, Ok, Result

from .functional import ErrWithTraceback

# Types

# Don't rely on the generic type. Wrong annotation might be missed.
# Use `Any` to signal that uncertainty explicitly.
# TODO [P1]: is this useful?
NPA = npt.NDArray[Any]

ArrayLike = npt.ArrayLike


T = TypeVar("T")


TR = TypeVar("TR", covariant=True)
E = TypeVar("E", covariant=True)


class Record(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)

    def __repr__(self) -> str:
        return json.dumps(self.model_dump(), indent=2)

    def __str__(self) -> str:
        return self.__repr__()

    def to_dict(self):
        return self.model_dump()


# Define a JSONValue as a union of common JSON value types

JSONPrimitive = str | int | bool | float | None

JSONDict = dict[str, "JSONValue"]
JSONList = list["JSONValue"]

JSONValue = JSONPrimitive | JSONList | JSONDict

JSONObject = JSONDict


P = ParamSpec("P")


F = TypeVar("F", bound=Callable[..., Any])
U_Callable = TypeVar("U_Callable", bound=Callable[..., Any])


LOGGER_FMT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d: %(message)s"


def get_logger(
    module_name: str,
    log_level: str | int = logging.WARNING,
    log_file_path: str = "log.txt",
) -> logging.Logger:
    logging.basicConfig(format=LOGGER_FMT)

    logger = logging.getLogger(module_name)
    logger.setLevel(log_level)
    log_handler = logging.FileHandler(log_file_path, mode="a")
    formatter = logging.Formatter(LOGGER_FMT)
    log_handler.setFormatter(formatter)

    logger.addHandler(log_handler)
    return logger


T2 = TypeVar("T2")
U2 = TypeVar("U2")
U = TypeVar("U")


Thunk = Callable[[], T]
ResultThunk = Callable[[], Result[T, E]]

G = TypeVar("G", bound=Callable[..., Any])
Decorator = Callable[[F], G]


T_Output = TypeVar("T_Output", covariant=True)


def read_file_from_handle(f_handle: IO[str]) -> Result[str, str]:
    try:
        return Ok(f_handle.read())
    except IOError as e:
        return ErrWithTraceback(e)


def read_text_file(path: str | None) -> Result[str, str]:
    """Read a file from a path. Returns the contents of the file as a string or an Err.
    Does not raise.
    """
    path_fn = make_safe_file_io_fn(read_file_from_handle, "r")
    return path_fn(path)


def write_text_file(path: str | None, contents: str) -> Result[int, str]:
    """Write contents to a file path. Returns the write result if successful, or an Err.
    Does not raise.
    """

    def _write(f_handle: IO[str]) -> Result[int, str]:
        return write_to_text_file_handle(f_handle, contents)

    path_fn = make_safe_file_io_fn(_write, "w")
    return path_fn(path)


def write_to_text_file_handle(
    f_handle: IO[str], contents: str
) -> Result[int, str]:
    try:
        return Ok(f_handle.write(contents))
    except IOError as e:
        return ErrWithTraceback(e)


def make_safe_file_io_fn(
    f_handle_fn: Callable[[IO[str]], Result[T, str]],
    mode: str,
    encoding: str | None = None,
) -> Callable[[str | None], Result[T, str]]:
    """Helper function to abstract out the logic of interacting with a file
        while handling IOError. Given a function `f_handle_fn` that operators on a file handle,
        this decorator will return a new function
        that takes a file path and runs `f_handle_fn` on that file.


    Args:
        f_handle_fn (Callable[[IO[str]], Result[T, str]]):
            Function that operates on an open file handle (i.e. read/write)
        mode (str): same as `open()` mode (e.g. "r", "w", "a")
        encoding (str | None, optional): Same as `open()` encoding. Defaults to "utf8".

    Returns:
        Callable[[str | None], Result[T, str]]: New function that does the same logic
            as `f_handle_fn`but take a a file path as input instead of a file handle.
    """
    encoding = encoding or "utf8"

    def _text_path_handler(path: str | None) -> Result[T, str]:
        if not path:
            return Err("Path is None")
        try:
            with open(path, mode, encoding=encoding) as f:
                return f_handle_fn(f)
        except IOError as e:
            return ErrWithTraceback(e)

    return _text_path_handler


def flatten_list(list_of_lists: list[list[T]]) -> list[T]:
    return [item for sublist in list_of_lists for item in sublist]


def unflatten_iterable(it: Iterable[T], chunk_size: int) -> list[list[T]]:
    """Inverse of flatten_list. Chunks an iterable into a 2d list. Output can be jagged, not padded."""
    out: list[list[T]] = [[]]
    for x in it:
        if len(out[-1]) == chunk_size:
            out.append([])
        out[-1].append(x)

    return out


def exp_backoff(
    max_retries: int,
    base_delay: int,
    logger: Optional[logging.Logger] = None,
) -> Decorator[Thunk[U], ResultThunk[U, str]]:
    logger = logger or logging.getLogger(__name__)

    def dec(thunk: Thunk[U]) -> ResultThunk[U, str]:
        def wrapper_thunk() -> Result[U, str]:
            retries = 0
            while retries < max_retries:
                try:
                    return Ok(thunk())
                except Exception as e:
                    logger.info(f"Attempt {retries + 1} failed: {e}")
                    delay = base_delay * 2**retries
                    logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    retries += 1

            return Err("Max retries reached, operation failed.")

        return wrapper_thunk

    return dec


def remove_nones(d: Mapping[str, Any]) -> dict[str, Any]:
    """Helper function that just remove None values. This is useful
    when you want to distinguish a missing value from an explicit user-supplied `None`.
    E.g. you can call it immediately after parsing argparse args."""
    return {k: v for k, v in d.items() if v is not None}


def dict_union_allow_replace(
    *dicts: Mapping[T, Any], on_conflict: Optional[str] = None
) -> dict[T, Any]:
    """Convenience function to combine dictionaries with overwrites on conflicts

    Args:
        on_conflict (Optional[str], optional): "keep_first" | "replace".

    Returns:
        dict[T, Any]: _description_
    """
    assert on_conflict != "err"
    res = dict_union(*dicts, on_conflict=on_conflict)
    match res:
        case Ok(d):
            return d
        case Err(e):
            assert False, f"should be unreachable: {e=}"


def dict_union(
    *dicts: Mapping[T, Any], on_conflict: Optional[str] = None
) -> Result[dict[T, Any], str]:
    """Like set union, but for dicts.

    Args:
        on_conflict (Optional[str], optional): what to do if two dicts have the same key with different values.
        Choices: "keep_first" | "replace" | "err". Defaults to "err".

    Returns:
        Result[dict[T, Any], str]: combined dict or Err if there is a conflict and on_conflict=="err".
    """
    # TODO make this an enum
    on_conflict = on_conflict or "err"

    result = {}
    for d in dicts:
        for k, v in d.items():
            if k not in result or result[k] == v:
                result[k] = v
            else:
                if on_conflict == "err":
                    return Err(
                        f"Key {k} exists with different values: {v}, {result[k]}."
                    )
                elif on_conflict == "keep_first":
                    pass
                elif on_conflict == "replace":
                    result[k] = v
                else:
                    assert (
                        False
                    ), f"should be unreachable: invalid {on_conflict=}"

    return Ok(result)


def enforce_unique(iterable: Iterable[T]) -> Result[T, str]:
    """Helper function that asserts an iterable contains only one unique value.
    This is useful when aggregating denormalized data."""
    iter_ = iter(iterable)
    value = next(iter_)
    for x in iter_:
        if value != x:
            return Err(f"distinct values: {value}, {x}")
    return Ok(value)


only = enforce_unique


def is_unique(iterable: Iterable[T]) -> bool:  # type: ignore[no-untyped-def]
    return enforce_unique(iterable).is_ok()


def dict_map(
    d: Mapping[T, U],
    key_fn: Optional[Callable[[T, U], T2]] = None,
    value_fn: Optional[Callable[[T, U], U2]] = None,
) -> dict[T2, U2]:
    """Like list map, but for dicts.

    Args:
        d (Mapping[T, U]): any Mapping
        key_fn (Optional[Callable[[T, U], T2]], optional): Function to map over the keys.
            Accepts both the key and value as argument, and must return the new key.
        value_fn (Optional[Callable[[T, U], U2]], optional): _description_. Function to map over the values.
            Accepts both the key and value as argument, and must return the new value.

    Returns:
        dict[T2, U2]: New dict with mapped keys and values.
    """

    def key_fn_(k: T, v: U) -> T2:
        if key_fn:
            return key_fn(k, v)
        else:
            return cast(T2, k)

    def value_fn_(k: T, v: U) -> U2:
        if value_fn:
            return value_fn(k, v)
        else:
            return cast(U2, v)

    return {key_fn_(k, v): value_fn_(k, v) for k, v in d.items()}


def dict_invert(d: Mapping[T, U]) -> Result[dict[U, T], str]:
    """Input: d is a one-to-one mapping. Output: Inverse mapping."""
    out = {}
    for k, v in d.items():
        if v in out:
            return Err(f"Not one-to-one: {k=},{v=}, {out[v]=}")
        else:
            out[v] = k

    return Ok(out)


def combine_returncodes(returncodes: Iterable[int]) -> int:
    """
    Return 0 if all 0 or first non-zero return code
    """
    for rc in returncodes:
        if rc != 0:
            return rc

    return 0


def load_json(json_str: str) -> Result[JSONObject, str]:
    """More powerful loader than `json` module. Supports comments and trailing commas.
    Input: serialized JSON
    Returns: deserialized JSON or Err if invalid"""
    parser = JsonComment()
    try:
        loaded = cast(JSONObject, parser.loads(json_str))  # type: ignore
        return Ok(loaded)
    except json.JSONDecodeError as e:
        return ErrWithTraceback(e)


def load_json_file(file_path: str | None) -> Result[JSONObject, str]:
    """More powerful loader than `json` module. Supports comments and trailing commas.
    Input: path to JSON file
    Returns: deserialized JSON or Err if invalid"""
    contents = read_text_file(file_path)
    return contents.and_then(load_json)


def normalize_path(path: str) -> str:
    return os.path.realpath(path)


def deprefix(s: str, pfx: str) -> str:
    if s.startswith(pfx):  # Checks if the string starts with the given prefix
        return s[len(pfx) :]  # If true, returns the string without the prefix
    else:
        return s


T_Basemodel = TypeVar("T_Basemodel", bound=BaseModel)


def safe_model_validate_json_object(
    cls: Type[T_Basemodel], data: JSONObject
) -> Result[T_Basemodel, str]:
    try:
        return Ok(cls.model_validate(data))
    except ValidationError as e:
        return ErrWithTraceback(e)


def unzip(l: Sequence[tuple[T, U]]) -> tuple[list[T], list[U]]:
    return tuple(map(list, zip(*l)))  # type: ignore[fixme, return-value]


def print_result(
    r: Result[T, str], to: str | IO[str] | None = None
) -> Result[int, str]:
    data = fmt_result(r)
    match to:
        case None:
            return print_result(r, sys.stdout)
        case IO():
            return write_to_text_file_handle(to, data)
        case "stderr":
            print(r, file=sys.stderr)
            return Ok(0)
        case "stdout":
            print(r, file=sys.stdout)
            return Ok(0)
        case _:
            return write_text_file(to, data)


def fmt_result(r: Result[T, str]) -> str:
    match r:
        case Ok(value):
            return f"Ok:\n" + str(value) + "\n"
        case Err(msg):
            return f"Err:\n" + msg + "\n"


T_BaseModel = TypeVar("T_BaseModel", bound=BaseModel)


def safe_model_validate_json(
    s: str, basemodel_type: Type[T_BaseModel]
) -> Result[T_BaseModel, str]:
    try:
        return Ok(basemodel_type.model_validate_json(s))
    except ValueError as e:
        return ErrWithTraceback(e)


def pydantic_model_validate_from_json_file_handle(
    f_handle: IO[str], basemodel_type: Type[T_BaseModel]
) -> Result[T_BaseModel, str]:
    return result.do(
        safe_model_validate_json(
            file_contents_ok, basemodel_type=basemodel_type
        )
        for file_contents_ok in read_file_from_handle(f_handle)
    )


def pydantic_model_validate_from_json_file_path(
    path: str, basemodel_type: Type[T_BaseModel]
) -> Result[T_BaseModel, str]:
    fn = make_safe_file_io_fn(
        partial(
            pydantic_model_validate_from_json_file_handle,
            basemodel_type=basemodel_type,
        ),
        "r",
    )

    return fn(path)


def hash_id(data: Any) -> str:
    return sha256(str(data).encode("utf-8")).hexdigest()


async def run_thunk_safe(
    thunk: Coroutine[Any, Any, T], timeout: int
) -> Result[T, str]:
    try:
        task = asyncio.create_task(thunk)
        res = await asyncio.wait_for(task, timeout=timeout)
        return Ok(res)
    except BaseException as e:  # type: ignore
        # TODO [P1] log
        return Err(str(e))
