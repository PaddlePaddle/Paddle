# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import difflib
import re
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar, Union

from typing_extensions import TypeAlias

T = TypeVar("T")
E = TypeVar("E", bound=Exception)

REGEX_API_NAME = re.compile(r"(?P<api_full_name>[a-zA-Z0-9_\.]+)")
REGEX_ARGSPEC = re.compile(
    r"ArgSpec\((args=(?P<args>.*), varargs=(?P<varargs>.*), varkw=(?P<varkw>.*), defaults=(?P<defaults>.*), kwonlyargs=(?P<kwonlyargs>.*), kwonlydefaults=(?P<kwonlydefaults>.*), annotations=(?P<annotations>\{.*\}))?\)"
)
REGEX_WHITESPACE = re.compile(r"\s+")
REGEX_DOCUMENT = re.compile(r"\('document', '([0-9a-z]{32})'\)")


class Ok(Generic[T]):
    def __init__(self, value: T):
        self._value = value

    def ok(self):
        return self._value

    def err(self):
        raise ValueError("Ok.err")

    def is_ok(self):
        return True

    def is_err(self):
        return False

    def __repr__(self):
        return f"Ok({self._value})"


class Err(Generic[E]):
    def __init__(self, value: E):
        self._value = value

    def ok(self):
        raise ValueError("Err.ok")

    def err(self):
        return self._value

    def is_ok(self):
        return False

    def is_err(self):
        return True

    def __repr__(self):
        return f"Err({self._value})"


Result: TypeAlias = Union[Ok[T], Err[E]]
WithRemaining: TypeAlias = tuple[T, str]


@dataclass
class ArgSpec:
    args: str | None
    varargs: str | None
    varkw: str | None
    defaults: str | None
    kwonlyargs: str | None
    kwonlydefaults: str | None
    annotations: str | None

    def format(self, skip_fields: list[str]):
        formatted = []
        fields = self.__dataclass_fields__.keys()
        for field in fields:
            if field in skip_fields:
                continue
            value = getattr(self, field)
            if value is not None:
                formatted.append(f"{field}={value}")
        return f"ArgSpec({', '.join(formatted)})"

    def __repr__(self):
        return self.format([])


@dataclass
class Document:
    hash: str

    def format(self, skip_fields: list[str]):
        if "document" in skip_fields:
            return "('document', '**********')"
        return f"('document', '{self.hash}')"

    def __repr__(self):
        return self.format([])


@dataclass
class ApiSpec:
    name: str
    signature: ArgSpec
    document: Document

    def format(self, skip_fields: list[str]):
        return f"{self.name} ({self.signature.format(skip_fields)}) ({self.document.format(skip_fields)})"

    def __repr__(self):
        return self.format([])


class ParseError(Exception):
    pass


def eat_string(
    input: str, to_eat: str
) -> Result[WithRemaining[str], ParseError]:
    if input.startswith(to_eat):
        return Ok((to_eat, input[len(to_eat) :]))
    return Err(ParseError(f"Expected {to_eat} but got {input}"))


def eat_whitespace(input: str) -> Result[WithRemaining[str], ParseError]:
    match = REGEX_WHITESPACE.match(input)
    if match is None:
        return Err(ParseError(f"Expected whitespace but got {input}"))
    return Ok((match.group(), input[match.end() :]))


def parse_api_name(api: str) -> Result[WithRemaining[str], ParseError]:
    match = REGEX_API_NAME.match(api)
    if match is None:
        return Err(ParseError(f"Failed to parse API name from {api}"))
    return Ok((match.group("api_full_name"), api[match.end() :]))


def parse_arg_spec(arg_spec: str) -> Result[WithRemaining[ArgSpec], ParseError]:
    match = REGEX_ARGSPEC.match(arg_spec)
    if match is None:
        return Err(ParseError(f"Failed to parse ArgSpec from {arg_spec}"))
    match.groupdict()
    return Ok(
        (
            ArgSpec(
                match.groupdict().get("args"),
                match.groupdict().get("varargs"),
                match.groupdict().get("varkw"),
                match.groupdict().get("defaults"),
                match.groupdict().get("kwonlyargs"),
                match.groupdict().get("kwonlydefaults"),
                match.groupdict().get("annotations"),
            ),
            arg_spec[match.end() :],
        )
    )


def parse_document(
    document: str,
) -> Result[WithRemaining[Document], ParseError]:
    match = REGEX_DOCUMENT.match(document)
    if match is None:
        return Err(ParseError(f"Failed to parse Document from {document}"))
    return Ok((Document(match.group(1)), document[match.end() :]))


def try_result(res: Result[T, E]) -> T:
    if isinstance(res, Err):
        raise res.err()
    return res.ok()


def parse_signature(sig: str):
    sig = sig.strip()
    api_name, remaining = try_result(parse_api_name(sig))
    _, remaining = try_result(eat_whitespace(remaining))
    _, remaining = try_result(eat_string(remaining, "("))
    arg_spec, remaining = try_result(parse_arg_spec(remaining))
    _, remaining = try_result(eat_string(remaining, ","))
    _, remaining = try_result(eat_whitespace(remaining))
    document, remaining = try_result(parse_document(remaining))
    _, remaining = try_result(eat_string(remaining, ")"))
    api_spec = ApiSpec(api_name, arg_spec, document)
    return api_spec


def read_spec(file_path) -> list[str]:
    with open(file_path, 'r') as f:
        spec = f.read()
        spec = spec.splitlines()
    return spec


def cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check API compatibility")
    parser.add_argument("origin", type=str, help="Origin API file path")
    parser.add_argument("new", type=str, help="New API file path")
    parser.add_argument(
        "--skip-fields",
        type=str,
        default="",
        help="Skip fields in diff, separated by comma, e.g. 'args,varargs'",
    )
    return parser.parse_args()


def create_preprocessor(
    skip_fields: list[str],
) -> Callable[[list[str]], list[str]]:
    def line_preprocessor(spec: str) -> str:
        print(f"parsing {spec}")
        sig = parse_signature(spec)
        print(f"parsed {sig}")
        return sig.format(skip_fields)

    def api_preprocessor(spec: list[str]) -> list[str]:
        return list(map(line_preprocessor, spec))

    return api_preprocessor


def main():
    args = cli()
    origin_spec = read_spec(args.origin)
    new_spec = read_spec(args.new)
    preprocessor = create_preprocessor(args.skip_fields.split(','))

    differ = difflib.Differ()
    result = differ.compare(preprocessor(origin_spec), preprocessor(new_spec))

    error = False
    diffs = []
    for each_diff in result:
        if each_diff[0] in ['-', '?']:  # delete or change API is not allowed
            error = True
        elif each_diff[0] == '+':
            error = True

        if each_diff[0] != ' ':
            diffs.append(each_diff)

    if error:
        print('API Difference is: ')
        for each_diff in diffs:
            print(each_diff)


"""
If you modify/add/delete the API files, including code and comment,
please follow these steps in order to pass the CI:

1. cd ${paddle_path}, compile paddle;
2. pip install build/python/dist/(build whl package);
3. run "python tools/print_signatures.py paddle.base> paddle/fluid/API.spec"
"""
if __name__ == "__main__":
    main()
