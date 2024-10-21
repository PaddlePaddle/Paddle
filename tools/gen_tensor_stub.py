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
import importlib
import inspect
import logging
import re
import sys
import traceback
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol

from typing_extensions import TypeAlias, get_overloads

if TYPE_CHECKING:
    from types import ModuleType

logging.basicConfig(style="{", format="{message}", level=logging.INFO)
logger = logging.getLogger("Generating stub file for paddle.Tensor")
logger.setLevel(logging.INFO)

INDENT_SIZE = 4
INDENT = " " * INDENT_SIZE
MEANLESS_INDEX = -1


MemberType: TypeAlias = Literal[
    "doc",
    "attribute",
    "method",
]


class AnnoConverter(Protocol):
    """
    convert bad annotation, e.g.:

    "Literal[('raise', 'wrap', 'clip')]" -> "Literal['raise', 'wrap', 'clip']"
    """

    def convert(self, input: str) -> str: ...


class LiteralConverter(AnnoConverter):
    pattern = re.compile(
        r"(?P<lit_start>Literal\[)\((?P<content>.*?)\)(?P<lit_end>\])"
    )

    def convert(self, input: str) -> str:
        return self.pattern.sub(r'\g<lit_start>\g<content>\g<lit_end>', input)


@dataclass
class Member:
    id: int
    name: str
    type: MemberType
    aliases: list[str]
    decorators: list[str]
    signature: str
    doc: str | None

    def add_alias(self, alias: str):
        self.aliases.append(alias)


@lru_cache
def _slot_pattern(slot_name: str) -> re.Pattern:
    return re.compile(
        r"(?P<indent> *)#\s*annotation:\s*\$\{" + slot_name + r"\}"
    )


class TensorGen:
    _converters: list[AnnoConverter] = [LiteralConverter()]

    def __init__(self, template: str = '', prefix: str = 'tensor'):
        self._template = template
        self._template_codes: list[tuple[int, int, str]] = []
        self._prefix = prefix

    def find_annotation_slot(self, slot_name: str) -> tuple[str, int, int]:
        pattern = _slot_pattern(slot_name)
        slot = []
        for mo in pattern.finditer(self._template):
            _indent = mo.group('indent')
            _start_index, _end_index = mo.span()
            slot.append((_indent, _start_index, _end_index))

        assert len(slot) == 1, self._template
        return slot[0]

    @property
    def tensor_docstring(self):
        return self.find_annotation_slot(f'{self._prefix}_docstring')

    @property
    def tensor_attributes(self):
        return self.find_annotation_slot(f'{self._prefix}_attributes')

    @property
    def tensor_methods(self):
        return self.find_annotation_slot(f'{self._prefix}_methods')

    @property
    def tensor_alias(self):
        return self.find_annotation_slot(f'{self._prefix}_alias')

    @property
    def template_codes(self) -> list[tuple[int, int, str]]:
        return self._template_codes

    @property
    def template_begin(self) -> int:
        return self._template.find(f'{self._prefix}_begin')

    @property
    def template_end(self) -> int:
        return self._template.find(f'{self._prefix}_end')

    def find_apis(self, api_name: str) -> list[dict[str, tuple[str, int, int]]]:
        if self.template_begin < 0 or self.template_end < 0:
            return []

        pattern = re.compile(
            r"(?P<indent> *)(?P<def_api>def "
            + api_name
            + r")(?P<signature>\(.*?\).*?:)(?P<docstring>.*?)(?P<ellipsis>\.{3})(?P<comment>[^\n]*#[^\n]*\n)?",
            re.DOTALL,
        )
        api = []
        for mo in pattern.finditer(self._template):
            _start_index, _end_index = mo.span()
            if not (
                self.template_begin < _start_index < self.template_end
                and self.template_begin < _end_index < self.template_end
            ):
                continue

            _indent = mo.group('indent')
            _signature = mo.group('signature')
            _docstring = mo.group('docstring')
            _ellipsis = mo.group('ellipsis')
            _comment = mo.group('comment')
            _comment = '' if _comment is None else _comment

            _start_indent, _ = mo.span('indent')
            _start_docstring, _ = mo.span('docstring')
            _, _end_ellipsis = mo.span('ellipsis')
            _start_comment = _end_ellipsis
            _end_comment = _start_comment + len(_comment)

            assert _start_index == _start_indent
            assert _end_comment == _end_index

            _api = {
                'indent': (_indent, MEANLESS_INDEX, MEANLESS_INDEX),
                'signature': (_signature, MEANLESS_INDEX, MEANLESS_INDEX),
                'docstring': (_docstring, _start_docstring, _end_comment),
                'ellipsis': (_ellipsis, MEANLESS_INDEX, MEANLESS_INDEX),
                'comment': (_comment, MEANLESS_INDEX, MEANLESS_INDEX),
            }
            api.append(_api)

        return api

    def insert_template(self, code: str, start: int, end: int) -> None:
        if start != MEANLESS_INDEX and end != MEANLESS_INDEX:
            self._template_codes.append((start, end, code))

    def add_method(self, func: Member):
        """
        1. insert docstring: tensor.prototype.pyi define the method without docstring
        2. insert method: tensor.prototype.pyi NOT define the method
        """
        methods = self.find_apis(func.name)
        if methods:
            # only use the last method
            method = methods[-1]
            # insert docstring if necessary
            if not method['docstring'][0].strip():
                doc = func.doc
                if doc:
                    comment = method['comment'][0]
                    doc_start = method['docstring'][1]
                    doc_end = method['docstring'][2]

                    api_indent = method['indent'][0]

                    assert len(api_indent) % INDENT_SIZE == 0

                    _indent = api_indent + INDENT

                    _doc = '\n'  # new line
                    _doc += f'{_indent}r"""\n'
                    _doc += with_indent(doc, len(_indent) // INDENT_SIZE)
                    _doc += "\n"
                    _doc += f'{_indent}"""\n'
                    _doc += f'{_indent}...\n'
                    _doc += f'{_indent}\n'

                    self.insert_template(comment + _doc, doc_start, doc_end)
        else:
            method_code = '\n'
            for decorator in func.decorators:
                method_code += f"@{decorator}\n"

            method_code += f"def {func.signature}:\n"
            # do NOT insert docs from overload methods,
            # because we always add a plain method
            if func.doc and func.decorators != ["overload"]:
                method_code += f'{INDENT}r"""\n'
                method_code += with_indent(func.doc, 1)
                method_code += "\n"
                method_code += f'{INDENT}"""\n'
            method_code += f"{INDENT}...\n"

            _indent, _, _end_index = self.tensor_methods
            method_code = with_indent(method_code, len(_indent) // INDENT_SIZE)
            self.insert_template(method_code, _end_index, _end_index)

    def add_alias(self, alias: str, target: str):
        _indent, _, _end_index = self.tensor_alias
        aliases_code = "\n"
        aliases_code += f"{_indent}{alias} = {target}"
        self.insert_template(aliases_code, _end_index, _end_index)

    def add_attribute(self, name: str, type_: str):
        _indent, _, _end_index = self.tensor_attributes
        attr_code = "\n"
        attr_code += f"{_indent}{name}: {type_}"
        self.insert_template(attr_code, _end_index, _end_index)

    def add_doc(self, doc: str):
        _indent, _, _end_index = self.tensor_docstring
        docstring = "\n"
        docstring += 'r"""\n'
        docstring += doc
        docstring += "\n"
        docstring += '"""\n'
        docstring = with_indent(docstring, len(_indent) // INDENT_SIZE)
        self.insert_template(docstring, _end_index, _end_index)

    @classmethod
    def codegen(
        cls, template: str, template_codes: list[tuple[int, int, str]]
    ) -> str:
        header = (
            '# This file is auto generated by `tools/gen_tensor_stub.py`.\n\n'
        )

        _template = []
        start = 0
        end = 0
        for _start, _end, code in sorted(template_codes):
            end = _start
            _template.extend(
                [
                    template[start:end],
                    code,
                ]
            )
            start = _end
        _template.append(template[start:])

        _content = header + ''.join(_template)

        for converter in cls._converters:
            _content = converter.convert(_content)

        return _content


def is_inherited_member(name: str, cls: type) -> bool:
    """Check if the member is inherited from parent class"""

    # keep magic methods
    if name.startswith("__") and name.endswith("__"):
        return False

    if name in cls.__dict__:
        return False

    for base in cls.__bases__:
        if name in base.__dict__:
            return True

    return any(is_inherited_member(name, base) for base in cls.__bases__)


def is_property(member: Any) -> bool:
    """Check if the member is a property"""

    return isinstance(member, (property, cached_property))


def is_staticmethod(member: Any) -> bool:
    """Check if the member is a staticmethod"""

    return isinstance(member, staticmethod)


def is_classmethod(member: Any) -> bool:
    """Check if the member is a classmethod"""

    return isinstance(member, classmethod)


def process_lines(code: str, callback: Callable[[str], str]) -> str:
    lines = code.splitlines()
    end_with_newline = code.endswith("\n")
    processed_lines: list[str] = []
    for line in lines:
        processed_lines.append(callback(line))
    processed_code = "\n".join(processed_lines)
    if end_with_newline:
        processed_code += "\n"
    return processed_code


def with_indent(code: str, level: int) -> str:
    def add_indent_line(line: str) -> str:
        if not line:
            return line
        return INDENT + line

    def remove_indent_line(line: str) -> str:
        if not line:
            return line
        elif line.startswith(INDENT):
            return line[len(INDENT) :]
        else:
            return line

    if level == 0:
        return code
    elif level > 0:
        if level == 1:
            return process_lines(code, add_indent_line)
        code = process_lines(code, add_indent_line)
        return with_indent(code, level - 1)
    else:
        if level == -1:
            return process_lines(code, remove_indent_line)
        return with_indent(code, level - 1)


def func_sig_to_method_sig(func_sig: str) -> str:
    regex_func_sig = re.compile(
        r"^(?P<method_name>[_a-zA-Z0-9]+)\((?P<arg0>[^,*)]+(:.+)?)?(?P<rest_args>.*)?\)",
        re.DOTALL,
    )
    matched = regex_func_sig.search(func_sig)
    if matched is None:
        # TODO: resolve this case
        logging.warning(f"Cannot parse function signature: {func_sig}")
        return "_(self)"

    if matched.group('rest_args').startswith('*'):
        method_sig = regex_func_sig.sub(
            r"\g<method_name>(self, \g<rest_args>)", func_sig
        )

    else:
        method_sig = regex_func_sig.sub(
            r"\g<method_name>(self\g<rest_args>)", func_sig
        )

    return method_sig


def func_doc_to_method_doc(func_doc: str) -> str:
    # Iterate every line, insert the indent and remove document of the first argument
    method_doc = ""
    is_first_arg = False
    first_arg_offset = 0

    for line in func_doc.splitlines():
        current_line_offset = len(line) - len(line.lstrip())
        # Remove the first argument (self in Tensor method) from docstring
        if is_first_arg:
            if current_line_offset <= first_arg_offset:
                is_first_arg = False
            if not first_arg_offset:
                first_arg_offset = current_line_offset
            if is_first_arg:
                continue
        method_doc += f"{line}\n" if line else "\n"
        if line.lstrip().startswith("Args:"):
            is_first_arg = True

    return method_doc


def try_import_paddle() -> ModuleType | None:
    try:
        return importlib.import_module('paddle')
    except ModuleNotFoundError:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.write(
            '''
ERROR: Can NOT import paddle from `tools/gen_tensor_stub.py` before installation.
    So the stub file `python/paddle/tensor/tensor.pyi` of `paddle.Tensor` may be lost.
    We COULD import paddle without installation with all libs (.dll or .so) copied into dir `paddle/libs`,
    or path already been set for the system. Try the following steps to locate the problem.

    1. Build with `SKIP_STUB_GEN=ON make -j$(nproc)`.
    2. Install the wheel from `build/python/dist`.
    3. Try to `import paddle` and check the problems.
            '''
        )
    return None


def get_tensor_members(module: str = 'paddle.Tensor') -> dict[int, Member]:
    paddle = try_import_paddle()
    if not paddle:
        raise (
            ModuleNotFoundError(
                'Can NOT import paddle from tools/gen_tensor_stub.py.'
            )
        )

    tensor_class = eval(module)

    members: dict[int, Member] = {}
    for name, member in inspect.getmembers(tensor_class):
        member_id = id(member)
        member_doc = inspect.getdoc(member)
        member_doc_cleaned = (
            func_doc_to_method_doc(inspect.cleandoc(member_doc))
            if member_doc is not None
            else None
        )
        try:
            sig = inspect.signature(member)
            # TODO: classmethod
            member_signature = f"{name}{sig}"

        except (TypeError, ValueError):
            member_signature = f"{name}()"

        if is_inherited_member(name, tensor_class):
            continue

        # Filter out private members except magic methods
        if name.startswith("_") and not (
            name.startswith("__") and name.endswith("__")
        ):
            continue

        if member_id in members:
            members[member_id].add_alias(name)
            continue

        if name == '__doc__':
            members[member_id] = Member(
                member_id,
                name,
                "doc",
                [],
                [],
                "__doc__",
                member,
            )
        elif is_property(member) or inspect.isdatadescriptor(member):
            members[member_id] = Member(
                member_id,
                name,
                "method",
                [],
                ["property"],
                f"{name}(self)",
                member_doc_cleaned,
            )
        elif is_classmethod(member):
            members[member_id] = Member(
                member_id,
                name,
                "method",
                [],
                ["classmethod"],
                member_signature,
                member_doc_cleaned,
            )
        elif is_staticmethod(member):
            members[member_id] = Member(
                member_id,
                name,
                "method",
                [],
                ["staticmethod"],
                member_signature,
                member_doc_cleaned,
            )
        elif inspect.isfunction(member) or inspect.ismethod(member):
            # `all_signatures`： list[[member id, decorators, signature]]
            # with atleast an original method
            all_signatures = [[member_id, [], member_signature]]

            # try to get overloads
            _overloads = get_overloads(member)
            for f in _overloads:
                _sig = inspect.signature(f)
                all_signatures.append(
                    [
                        id(f),
                        ["overload"],
                        f"{name}{_sig}".replace("Ellipsis", "..."),
                    ]
                )

            for _member_id, _decorators, _sig in all_signatures:
                members[_member_id] = Member(
                    _member_id,
                    name,
                    "method",
                    [],
                    _decorators,
                    func_sig_to_method_sig(_sig),
                    member_doc_cleaned,
                )
        elif inspect.ismethoddescriptor(member):
            members[member_id] = Member(
                member_id,
                name,
                "method",
                [],
                [],
                func_sig_to_method_sig(member_signature),
                member_doc_cleaned,
            )
        else:
            logging.debug(f"Skip unknown type of member: {name}, {member}")

    return members


def get_tensor_template(path: str) -> str:
    with open(path) as f:
        return ''.join(f.readlines())


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        default="python/paddle/tensor/tensor.prototype.pyi",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="python/paddle/tensor/tensor.pyi",
    )

    args = parser.parse_args()

    return args


def generate_stub_file(input_file=None, output_file=None):
    # Get tensor template
    tensor_template = get_tensor_template(input_file)

    all_members: set[int] = set()
    template_codes: list[tuple[int, int, str]] = []
    # Get members of Tensor
    for module, prefix in [
        ['paddle.Tensor', 'tensor'],
        ['paddle.base.framework.EagerParamBase', 'eager_param_base'],
    ]:
        tensor_members = get_tensor_members(module)
        logging.debug(f'total members in {module}: {len(tensor_members)}')

        # Generate the Tensor stub
        tensor_gen = TensorGen(tensor_template, prefix)

        for member_id, member in tensor_members.items():
            if member_id in all_members:
                continue

            if member.type == "method":
                tensor_gen.add_method(member)
                for alias in member.aliases:
                    tensor_gen.add_alias(alias, member.name)
            elif member.type == "attribute":
                tensor_gen.add_attribute(member.name, "Any")
            elif member.type == "doc":
                tensor_gen.add_doc(member.doc)

        all_members |= tensor_members.keys()
        template_codes += tensor_gen.template_codes

    # Write to target file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(TensorGen.codegen(tensor_template, template_codes))


def main():
    args = parse_args()
    generate_stub_file(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
