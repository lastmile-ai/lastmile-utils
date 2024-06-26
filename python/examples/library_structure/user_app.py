import argparse
import my_library.api as my_library


class SomeUserException(Exception):
    pass


def use_library(key: str):
    print("Running example with key:", key)
    return my_library.example_api_function(key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key",
        required=True,
        choices=["good", "internal-problem", "user-problem"],
    )
    args = parser.parse_args()
    try:
        value = use_library(str(args.key))
        print("Value:", value)
    except Exception as e:
        raise SomeUserException() from e


if __name__ == "__main__":
    main()
