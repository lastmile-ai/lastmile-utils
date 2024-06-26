from lastmile_utils.lib.core.lm_result import (
    EResult,
    UserError,
    resultify_exceptions,
)


@resultify_exceptions
def _look_up(key: str) -> int:
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
        "good": 42,
    }

    value = mapping[key]
    return value


def impl_inner(key: str) -> EResult[int]:
    """Just some helper function"""
    if key == "user-problem":
        raise UserError("foo")
    else:
        return _look_up(key)


def impl_middle(key: str) -> EResult[int]:
    """Just some helper function"""
    return impl_inner(key)


def impl_outermost(key: str):
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
    return impl_middle(key)
