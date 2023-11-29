from typing import Any


class WrapObject:
    def __init__(
        self, real_object: Any, mock_key: str, mock_value: Any
    ) -> None:
        self.real_object = real_object
        self.mock_key = mock_key
        self.mock_value = mock_value

    def __getattr__(self, key: str) -> Any:
        if key == self.mock_key:
            return self.mock_value
        else:
            return getattr(self.real_object, key)

    def __repr__(self) -> str:
        return f"WrapObject(\n{self.real_object},\n{self.mock_key},\n{self.mock_value})"

    def __str__(self) -> str:
        return repr(self)


def make_wrap_object(
    real_object: Any, mock_path: str, mock_value: Any
) -> WrapObject:
    parts = mock_path.split(".")
    if len(parts) == 1:
        return WrapObject(real_object, parts[0], mock_value)
    else:
        first, rest = parts[0], ".".join(parts[1:])
        return WrapObject(
            real_object,
            first,
            make_wrap_object(getattr(real_object, first), rest, mock_value),
        )
