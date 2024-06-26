from typing import Optional
from lastmile_utils.lib.core.lm_result import (
    EResult,
    careful_do_you_really_want_to_call_me_return_or_raise,
)

from .lib import impl_outermost


def example_api_function(key: str, default_int: Optional[int]) -> int:
    """    
    This is an example LM API function.

    It is deliberately extremely short and simple,
    in the spirit of a header file. 
    
    It is a wrapper around the actual implementation,
    in this case `impl_outermost()` which is imported from
    our library.

    Notice that impl_outermost() returns an EResult,
    which is why we use the decorator here.

    The decorator not only automatically unpacks the value
    or raises an exception, but it also does some logging for us
    and specific handling of different common Exception types.
    """
    @careful_do_you_really_want_to_call_me_return_or_raise
    def impl_(key: str, default_int: Optional[int]) -> EResult[int]:
        output = impl_outermost(key, default_int)
        return output
    
    return impl_(key, default_int)
