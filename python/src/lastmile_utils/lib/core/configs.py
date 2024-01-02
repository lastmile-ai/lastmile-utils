import argparse
from dataclasses import dataclass
import logging
import os
from enum import Enum, EnumMeta
from types import UnionType
from typing import Any, Dict, Mapping, Optional, Set, Type, TypeVar
import typing

import pydantic
from dotenv import load_dotenv
from result import Ok, Result

from .utils import (
    ErrWithTraceback,
    JSONObject,
    Record,
    dict_union_allow_replace,
    remove_nones,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class TypeOrigin:
    origin: type
    args: list["TypeOrigin"]


def _alias_type_to_origin_type(
    some_type: Type[Any] | type,
) -> TypeOrigin | None:
    args = (
        []
        if not hasattr(some_type, "__args__")
        else [_alias_type_to_origin_type(a) for a in getattr(some_type, "__args__")]
    )
    args = [a for a in args if a is not None]
    constructor = getattr(some_type, "__origin__", some_type)
    return TypeOrigin(constructor, args)


def _get_inside_of_optional_type(
    some_type: Type[Any] | type,
) -> Optional[type]:
    """
    examples:
    - Optional[int] -> int
    - Optional[Optional[list[str]]] -> list
    - list[str] -> None
    """
    origin_type = _alias_type_to_origin_type(some_type)
    if origin_type is not None and origin_type.origin == typing.Union:
        type_args = getattr(some_type, "__args__")
        if not (len(type_args) == 2 and type_args[1] == type(None)):
            return None

        inside_type = type_args[0]
        if not hasattr(inside_type, "__origin__"):
            return inside_type
        else:
            return getattr(inside_type, "__origin__")


def _parse_log_level(x: int | str) -> int:
    match x:
        case int(x_i):
            return x_i
        case str(x_s):
            out = logging.getLevelName(x_s.strip().upper())
            match out:
                case int(out_i):
                    return out_i
                case _:
                    msg = "Choices: ['CRITICAL', 'FATAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']"
                    raise argparse.ArgumentTypeError(msg)


def add_parser_argument(
    parser: argparse.ArgumentParser,
    field_name: str,
    field: pydantic.fields.FieldInfo,
    is_required: bool,
) -> Optional[str]:
    return add_parser_argument_helper(parser, field_name, field.annotation, is_required)


def add_parser_argument_helper(
    parser: argparse.ArgumentParser,
    field_name: str,
    field_type: Type[Any] | type | None,
    is_required: bool,
) -> Optional[str]:
    field_name = field_name.replace("_", "-")
    inside_of_optional_type = (
        _get_inside_of_optional_type(field_type) if field_type is not None else None
    )
    if inside_of_optional_type is not None:
        return add_parser_argument_helper(
            parser, field_name, inside_of_optional_type, False
        )
    elif field_name in ["log-level"]:
        parser.add_argument(
            f"--{field_name}",
            type=_parse_log_level,
            required=is_required,
        )
    elif field_type is None:
        return f"{field_name}: type is None"
    elif field_type is bool:
        if is_required:
            return f"(bool) flag cannot be required: field={field_name}"
        parser.add_argument(
            f"--{field_name}",
            action="store_true",
        )
    elif type(field_type) is UnionType:
        return f"UnionType not supported. If your field is a `type | None`, use Optional[type] instead. field={field_name}"
    elif isinstance(field_type, EnumMeta):
        parser.add_argument(
            f"--{field_name}",
            type=str,
            choices=[e.value.lower() for e in field_type],  # type: ignore
            required=is_required,
        )
    elif field_type is list:
        parser.add_argument(
            f"--{field_name}",
            nargs="+",
            required=is_required,
        )
    else:
        parser.add_argument(f"--{field_name}", type=field_type, required=is_required)


def add_parser_arguments(
    parser: argparse.ArgumentParser,
    fields: dict[str, pydantic.fields.FieldInfo],
    required: Optional[Set[str]] = None,
):
    required = required or {f_name for f_name, f in fields.items() if f.is_required()}
    for field_name, field in fields.items():
        is_required = field_name in required
        res = add_parser_argument(parser, field_name, field, is_required)
        if res is not None:
            LOGGER.warning(res)


def argparsify(
    r: Record | Type[Record],
    required: Optional[Set[str]] = None,
    subparser_record_types: Mapping[str, Record | Type[Record]] = {},
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_parser_arguments(parser, r.model_fields, required=required)

    if subparser_record_types:
        subparsers = parser.add_subparsers(dest="subcommand", required=True)
        for subparser_name, subparser_r in subparser_record_types.items():
            subparser = subparsers.add_parser(subparser_name)
            add_parser_arguments(subparser, subparser_r.model_fields)

    return parser


def get_config_args(args: argparse.Namespace | Dict[str, Any]) -> Dict[str, Any]:
    def _get_cli():
        if isinstance(args, argparse.Namespace):
            return vars(args)
        else:
            return dict(args)

    env_keys = {
        "openai_key": "OPENAI_API_KEY",
        "pinecone_key": "PINECONE_API_KEY",
        "pinecone_environment": "PINECONE_ENVIRONMENT",
        "pinecone_index_name": "PINECONE_INDEX_NAME",
        "pinecone_namespace": "PINECONE_NAMESPACE",
    }

    load_dotenv()
    env_values = remove_nones({k: os.getenv(v) for k, v in env_keys.items()})
    cli_values = remove_nones(_get_cli())

    all_args = dict_union_allow_replace(env_values, cli_values, on_conflict="replace")
    return all_args


def resolve_path(data_root: str, path: str) -> str:
    """
    data_root: path relative to cwd where all the data is stored.
    path: relative to data_root, the specific data to look at,
    e.g. "10ks/".
    """

    joined = os.path.join(os.getcwd(), data_root, path)
    return os.path.abspath(os.path.expanduser(joined))


T_Record = TypeVar("T_Record", bound=Record)


def get_subparser_name(main_parser: argparse.ArgumentParser, argv: list[str]) -> str:
    parsed = main_parser.parse_args(argv)
    return parsed.subcommand


def parse_args(
    parser: argparse.ArgumentParser,
    argv: list[str],
    config_type: Type[T_Record],
) -> Result[T_Record, str]:
    parsed = parser.parse_args(argv)
    cli_dict = remove_nones(vars(parsed))
    return config_from_primitives(config_type, cli_dict)


def config_from_primitives(
    config_type: Type[T_Record], cli_dict: JSONObject
) -> Result[T_Record, str]:
    try:
        return Ok(config_type.model_validate(cli_dict))
    except pydantic.ValidationError as e:
        return ErrWithTraceback(e, "Invalid args")


T_Enum = TypeVar("T_Enum", bound=Enum)


def validate_enum_field_value(value: Any, field_type: Type[T_Enum]) -> T_Enum:
    if isinstance(value, str):
        try:
            out = field_type[value.upper()]
            return out
        except KeyError as e:
            raise ValueError(
                f"Unexpected value for {field_type.__name__}: {value}"
            ) from e
    elif not isinstance(value, field_type):
        raise ValueError(
            f"Unexpected type for {field_type.__name__}: {value}, {type(value)}"
        )
    else:
        return value


class EnumValidatedRecordMixin(pydantic.BaseModel):
    @pydantic.root_validator(pre=True)
    def check_enum_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for field_name, value in values.items():
            if field_name not in cls.model_fields:
                # This can happen if an extra field is given.
                # In this case, if all the other fields are valid,
                # Pydantic actually allows this and ignores the extra field.
                # Here, we're just checking Enum fields, so we can ignore this
                # field and continue to validate the other fields.
                continue
            field: pydantic.fields.FieldInfo = cls.model_fields[field_name]
            _type = field.annotation
            if isinstance(_type, EnumMeta):
                out[field_name] = validate_enum_field_value(value, _type)
            else:
                out[field_name] = value

        return out
