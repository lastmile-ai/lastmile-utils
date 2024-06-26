import argparse
from typing import Optional
import my_library.api as my_library


class UserAppDefinedException(Exception):
    pass


def use_library(key: str, default_int: Optional[int] = None):
    print("Running example with key:", key)
    return my_library.example_api_function(key, default_int)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key",
        required=True,
        type=str,
        choices=[
            "good",
            # uncaught exception raised
            "bug",
            # User error
            "bad-key",
            # Potentially Recoverable error
            "bad-value",
        ],
    )
    parser.add_argument("--default-int", type=int)
    args = parser.parse_args()
    try:
        value = use_library(args.key, args.default_int)
        print("Value:", value)
    except Exception as e:
        raise UserAppDefinedException() from e


if __name__ == "__main__":
    main()
