from functools import partial
import logging
from textwrap import dedent
from typing import Optional
from result import Err, Ok
from lastmile_utils.lib.core.lm_result import (
    EResult,
    make_err,
    resultify_exceptions,
)

import lastmile_utils.lib.core.utils as utils
from lastmile_utils.lib.core.utils import UserError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format=utils.LOGGER_FMT)


@resultify_exceptions
def _mapping_lookup(key: str) -> str:
    """This is the bottom of our call stack, and the possible source of
    an Exception.

    Specifically, we are calling into Python core library code
    by doing the mapping lookup below.

    Therefore, this is a good place to use the `resultify_exceptions` decorator.

    The decorator catches any Exception and wraps it in a Result type (Err).
    If we hit the return, it will be wrapped in an Ok.

    The idea is to quarantine unchecked problems in as tight a space as possible.

    If we hit an Exception, it will be caught immediately, and
    throughout the rest of the call stack it will be statically checkable.

    """
    mapping = {
        "good": "42",
        "bad-value": "invalid-for-some-reason",
    }

    value = mapping[key]
    return value


def _look_up_value(key: str) -> EResult[str]:
    """Just some helper function"""
    if key == "bug":
        # Deliberate example bug: exception raised.
        # Unfortunately, there is no way to find these statically.
        raise ValueError("bar")
    else:

        return _mapping_lookup(key)


def _transform_key_step_1(key: str) -> EResult[str]:
    """Just some helper function"""
    if key == "bad-key":
        return Err(UserError(f"Can't preprocess key {key}"))
    else:
        return Ok(key.strip())


def _transform_key_step_2(key: str) -> str:
    """Just some helper function"""
    return key.lower()


def _transform_value(value: str, default_int: Optional[int]) -> EResult[int]:
    transformed = _transform_value_helper(value)

    def _err_to_default_int(e: Exception) -> EResult[int]:
        if default_int is None:
            return Err(
                UserError(
                    dedent(
                        f"""
                        No default int provided, and failed to transform value 
                        due to:
                        Exception: {e}
                        """
                    )
                )
            )
        else:
            return Ok(default_int)

    # Use this to recover from Err in certain cases.
    return transformed.or_else(_err_to_default_int)


def _transform_value_helper(value: str) -> EResult[int]:
    if len(value) == len("invalid-for-some-reason"):
        return make_err(ValueError("value is invalid"))
    else:
        return Ok(len(value))


def impl_middle(key: str, default_int: Optional[int]) -> EResult[int]:
    """Just some helper function"""
    return (
        _transform_key_step_1(key)
        .map(_transform_key_step_2)
        .and_then(_look_up_value)
        .and_then(partial(_transform_value, default_int=default_int))
    )


def impl_outermost(key: str, default_int: Optional[int]) -> EResult[int]:
    """Outermost implementation function. Calls down into a stack of
    helper functions.

    Notice that most of this code is statically checkable for exceptions.
    Every function boundary in this file is wrapped in a Result type.
    Since we have used the `resultify_exceptions` decorator at the lowest point,
    we don't have to catch exceptions.

    Just pay attention to pyright and handle all the Result types in
    some statically safe way.
    If pyright is happy, you are much less likely to get hit with a bug report.
    """
    value = impl_middle(key, default_int)

    return value
