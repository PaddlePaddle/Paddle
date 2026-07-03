# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
import os
import re
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

ATEN_OPS_DIR = Path("paddle/phi/api/include/compat/ATen/ops")
ATEN_TENSOR_BODY = Path("paddle/phi/api/include/compat/ATen/core/TensorBody.h")
ATEN_TENSOR_BASE = Path("paddle/phi/api/include/compat/ATen/core/TensorBase.h")
TORCH_INSTALL_HINT = (
    "pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu"
)

BUILTIN_TYPE_WORDS = {
    "auto",
    "bool",
    "char",
    "double",
    "float",
    "int",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "long",
    "short",
    "size_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "void",
}


@dataclass(frozen=True)
class CppSignature:
    raw: str
    canonical: str
    name: str
    is_member: bool


@dataclass(frozen=True)
class CheckError:
    header: Path
    message: str
    signature: CppSignature | None = None
    torch_header: Path | None = None
    candidates: tuple[CppSignature, ...] = ()


class SignatureParseError(ValueError):
    pass


EXPORT_MACROS = (
    "AT_API",
    "C10_API",
    "CAFFE2_API",
    "TORCH_API",
)


def strip_comments_and_literals(text: str) -> str:
    result: list[str] = []
    i = 0
    state = "normal"
    while i < len(text):
        ch = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if state == "normal":
            if ch == "/" and nxt == "/":
                result.extend((" ", " "))
                i += 2
                state = "line_comment"
                continue
            if ch == "/" and nxt == "*":
                result.extend((" ", " "))
                i += 2
                state = "block_comment"
                continue
            if ch == '"':
                result.append(ch)
                i += 1
                state = "string"
                continue
            if ch == "'":
                result.append(ch)
                i += 1
                state = "char"
                continue
            result.append(ch)
            i += 1
            continue
        if state == "line_comment":
            result.append("\n" if ch == "\n" else " ")
            i += 1
            if ch == "\n":
                state = "normal"
            continue
        if state == "block_comment":
            if ch == "*" and nxt == "/":
                result.extend((" ", " "))
                i += 2
                state = "normal"
                continue
            result.append("\n" if ch == "\n" else " ")
            i += 1
            continue
        if state in ("string", "char"):
            result.append(ch)
            if ch == "\\" and nxt:
                result.append(nxt)
                i += 2
                continue
            if (state == "string" and ch == '"') or (
                state == "char" and ch == "'"
            ):
                state = "normal"
            i += 1
            continue
    return "".join(result)


def find_matching(text: str, open_pos: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    pos = open_pos
    state = "normal"
    while pos < len(text):
        ch = text[pos]
        nxt = text[pos + 1] if pos + 1 < len(text) else ""
        if state == "normal":
            if ch == '"':
                state = "string"
                pos += 1
                continue
            if ch == "'":
                state = "char"
                pos += 1
                continue
        elif state in ("string", "char"):
            if ch == "\\" and nxt:
                pos += 2
                continue
            if (state == "string" and ch == '"') or (
                state == "char" and ch == "'"
            ):
                state = "normal"
            pos += 1
            continue

        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return pos
        pos += 1
    raise SignatureParseError(
        f"unmatched {open_ch!r} at character offset {open_pos}"
    )


def normalize_punctuation(text: str) -> str:
    text = re.sub(r"\bat::", "", text)
    text = re.sub(r"\bc10::", "", text)
    text = re.sub(r"\bcaffe2::", "", text)
    text = text.replace("::std::", "std::")
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s*::\s*", "::", text)
    text = re.sub(r"\s*([<>,()=\[\]])\s*", r"\1", text)
    text = re.sub(r"\s*([&*])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_signature_text(raw: str) -> str:
    raw = re.sub(r"\\\s*\n", " ", raw)
    lines = [
        line for line in raw.splitlines() if not line.lstrip().startswith("#")
    ]
    text = " ".join(lines)
    text = re.sub(r"\b(public|protected|private)\s*:\s*", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.rstrip(";").strip()


def strip_leading_templates(text: str) -> tuple[tuple[str, ...], str]:
    templates: list[str] = []
    rest = text.strip()
    while rest.startswith("template"):
        match = re.match(r"template\s*<", rest)
        if not match:
            break
        open_pos = rest.find("<", match.start())
        close_pos = find_matching(rest, open_pos, "<", ">")
        templates.append(normalize_punctuation(rest[: close_pos + 1]))
        rest = rest[close_pos + 1 :].strip()
    return tuple(templates), rest


def find_function_paren(text: str) -> int:
    depth_angle = 0
    positions: list[int] = []
    for pos, ch in enumerate(text):
        if ch == "<":
            depth_angle += 1
        elif ch == ">" and depth_angle > 0:
            depth_angle -= 1
        elif ch == "(" and depth_angle == 0:
            positions.append(pos)
    if not positions:
        raise SignatureParseError("function signature has no parameter list")
    return positions[-1]


def split_top_level(text: str, separator: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    angle = paren = bracket = brace = 0
    pos = 0
    state = "normal"
    while pos < len(text):
        ch = text[pos]
        nxt = text[pos + 1] if pos + 1 < len(text) else ""
        if state == "normal":
            if ch == '"':
                state = "string"
                pos += 1
                continue
            if ch == "'":
                state = "char"
                pos += 1
                continue
        elif state in ("string", "char"):
            if ch == "\\" and nxt:
                pos += 2
                continue
            if (state == "string" and ch == '"') or (
                state == "char" and ch == "'"
            ):
                state = "normal"
            pos += 1
            continue

        if ch == "<":
            angle += 1
        elif ch == ">" and angle > 0:
            angle -= 1
        elif ch == "(":
            paren += 1
        elif ch == ")" and paren > 0:
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]" and bracket > 0:
            bracket -= 1
        elif ch == "{":
            brace += 1
        elif ch == "}" and brace > 0:
            brace -= 1
        elif (
            ch == separator
            and angle == 0
            and paren == 0
            and bracket == 0
            and brace == 0
        ):
            pieces.append(text[start:pos])
            start = pos + 1
        pos += 1
    pieces.append(text[start:])
    return pieces


def split_default_value(param: str) -> tuple[str, str | None]:
    angle = paren = bracket = brace = 0
    pos = 0
    state = "normal"
    while pos < len(param):
        ch = param[pos]
        nxt = param[pos + 1] if pos + 1 < len(param) else ""
        if state == "normal":
            if ch == '"':
                state = "string"
                pos += 1
                continue
            if ch == "'":
                state = "char"
                pos += 1
                continue
        elif state in ("string", "char"):
            if ch == "\\" and nxt:
                pos += 2
                continue
            if (state == "string" and ch == '"') or (
                state == "char" and ch == "'"
            ):
                state = "normal"
            pos += 1
            continue

        if ch == "<":
            angle += 1
        elif ch == ">" and angle > 0:
            angle -= 1
        elif ch == "(":
            paren += 1
        elif ch == ")" and paren > 0:
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]" and bracket > 0:
            bracket -= 1
        elif ch == "{":
            brace += 1
        elif ch == "}" and brace > 0:
            brace -= 1
        elif (
            ch == "="
            and angle == 0
            and paren == 0
            and bracket == 0
            and brace == 0
        ):
            return param[:pos], param[pos + 1 :]
        pos += 1
    return param, None


def strip_parameter_name(param_type: str) -> str:
    text = normalize_punctuation(param_type)
    if not text or text == "void":
        return text
    match = re.match(
        r"(?P<prefix>.+?)(?:\s+|(?<=[&*]))"
        r"(?P<name>[A-Za-z_]\w*)\s*(?P<array>(?:\[[^\]]*\])*)$",
        text,
    )
    if not match:
        match = re.match(
            r"(?P<prefix>.+?>)(?P<name>[A-Za-z_]\w*)"
            r"\s*(?P<array>(?:\[[^\]]*\])*)$",
            text,
        )
    if not match:
        return text
    prefix = match.group("prefix").rstrip()
    name = match.group("name")
    if name in BUILTIN_TYPE_WORDS or prefix.endswith("::"):
        return text
    return normalize_punctuation(prefix + match.group("array"))


def normalize_parameter(param: str, include_default: bool) -> str:
    left, default = split_default_value(param)
    normalized_type = strip_parameter_name(left)
    if include_default and default is not None:
        return f"{normalized_type}={normalize_punctuation(default)}"
    return normalized_type


def parse_cpp_signature(
    raw: str,
    class_member: bool = False,
    include_member_defaults: bool = False,
    class_name: str = "Tensor",
) -> CppSignature | None:
    text = clean_signature_text(raw)
    if not text or "(" not in text or ")" not in text:
        return None
    templates, text = strip_leading_templates(text)
    export_macro_pattern = "|".join(re.escape(macro) for macro in EXPORT_MACROS)
    text = re.sub(rf"\b(?:{export_macro_pattern})\b\s*", "", text)
    text = strip_leading_signature_annotations(text)
    text = re.sub(r"\binline\s+", "", text).strip()
    if class_member and re.match(
        rf"(?:explicit\s+)?~?{re.escape(class_name)}\s*\(", text
    ):
        return None
    try:
        open_pos = find_function_paren(text)
        close_pos = find_matching(text, open_pos, "(", ")")
    except SignatureParseError:
        return None
    before = text[:open_pos].strip()
    params = text[open_pos + 1 : close_pos]
    suffix = text[close_pos + 1 :].strip()
    name_match = re.search(
        r"(?P<name>(?:[A-Za-z_]\w*::)*[A-Za-z_]\w*)$", before
    )
    if not name_match:
        return None
    qualified_name = name_match.group("name")
    return_type = before[: name_match.start()].strip()
    if not return_type:
        return None
    member_prefixes = ("Tensor::", "TensorBase::")
    if "::" in qualified_name and not qualified_name.startswith(
        member_prefixes
    ):
        return None
    is_member = class_member or qualified_name.startswith(member_prefixes)
    if class_member and "::" not in qualified_name:
        qualified_name = f"{class_name}::{qualified_name}"
    name = qualified_name.split("::")[-1]
    include_default = include_member_defaults or not is_member
    param_items = []
    if params.strip() and params.strip() != "void":
        param_items = [
            normalize_parameter(param, include_default)
            for param in split_top_level(params, ",")
            if param.strip()
        ]
    canonical_return = normalize_punctuation(return_type)
    canonical_suffix = normalize_punctuation(suffix)
    canonical_templates = "".join(templates)
    canonical = (
        f"{canonical_templates}{canonical_return} "
        f"{qualified_name}({','.join(param_items)})"
    )
    if canonical_suffix:
        canonical = f"{canonical}{canonical_suffix}"
    return CppSignature(
        raw=normalize_punctuation(text),
        canonical=canonical,
        name=name,
        is_member=is_member,
    )


def strip_leading_signature_annotations(text: str) -> str:
    rest = text.strip()
    while True:
        original = rest
        while rest.startswith("[["):
            end = rest.find("]]")
            if end < 0:
                break
            rest = rest[end + 2 :].lstrip()
        for macro in ("C10_DEPRECATED_MESSAGE", "C10_DEPRECATED"):
            macro_match = re.match(rf"{macro}\b", rest)
            if not macro_match:
                continue
            rest = rest[macro_match.end() :].lstrip()
            if rest.startswith("("):
                close_pos = find_matching(rest, 0, "(", ")")
                rest = rest[close_pos + 1 :].lstrip()
            break
        if rest == original:
            return rest


def namespace_at_blocks(text: str) -> list[str]:
    return namespace_blocks(text, "at")


def namespace_blocks(text: str, namespace_name: str) -> list[str]:
    clean = strip_comments_and_literals(text)
    blocks: list[str] = []
    pattern = rf"\bnamespace\s+{re.escape(namespace_name)}\s*\{{"
    for match in re.finditer(pattern, clean):
        open_pos = clean.find("{", match.start())
        close_pos = find_matching(clean, open_pos, "{", "}")
        blocks.append(clean[open_pos + 1 : close_pos])
    return blocks


def top_level_function_definitions(
    block: str, include_member_defaults: bool = False
) -> list[CppSignature]:
    signatures: list[CppSignature] = []
    start = 0
    pos = 0
    angle = paren = bracket = brace = 0
    state = "normal"
    while pos < len(block):
        ch = block[pos]
        nxt = block[pos + 1] if pos + 1 < len(block) else ""
        if state == "normal":
            if ch == '"':
                state = "string"
                pos += 1
                continue
            if ch == "'":
                state = "char"
                pos += 1
                continue
        elif state in ("string", "char"):
            if ch == "\\" and nxt:
                pos += 2
                continue
            if (state == "string" and ch == '"') or (
                state == "char" and ch == "'"
            ):
                state = "normal"
            pos += 1
            continue

        if ch == "<":
            angle += 1
        elif ch == ">" and angle > 0:
            angle -= 1
        elif ch == "(":
            paren += 1
        elif ch == ")" and paren > 0:
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]" and bracket > 0:
            bracket -= 1
        elif ch == "{":
            if angle == 0 and paren == 0 and bracket == 0 and brace == 0:
                candidate = block[start:pos]
                signature = parse_cpp_signature(
                    candidate,
                    include_member_defaults=include_member_defaults,
                )
                if signature is not None:
                    signatures.append(signature)
                end = find_matching(block, pos, "{", "}")
                pos = end + 1
                start = pos
                continue
            brace += 1
        elif ch == "}" and brace > 0:
            brace -= 1
        if ch == ";":
            if angle == 0 and paren == 0 and bracket == 0 and brace == 0:
                candidate = block[start:pos]
                signature = parse_cpp_signature(
                    candidate,
                    include_member_defaults=include_member_defaults,
                )
                if signature is not None:
                    signatures.append(signature)
                start = pos + 1
        pos += 1
    return signatures


def parse_paddle_header(path: Path) -> tuple[list[CppSignature], list[str]]:
    blocks = namespace_at_blocks(path.read_text())
    errors: list[str] = []
    if len(blocks) not in (1, 2):
        errors.append(
            "expected one or two exact 'namespace at' blocks, "
            f"got {len(blocks)}"
        )
        return [], errors
    signatures: list[CppSignature] = []
    for block in blocks:
        signatures.extend(top_level_function_definitions(block))
    return signatures, errors


def parse_torch_ops_header(path: Path) -> list[CppSignature]:
    if not path.is_file():
        return []
    signatures: list[CppSignature] = []
    for block in namespace_at_blocks(path.read_text()):
        signatures.extend(
            sig
            for sig in top_level_function_definitions(block)
            if not sig.is_member
        )
        for symint_block in namespace_blocks(block, "symint"):
            signatures.extend(
                signature_without_templates(sig)
                for sig in top_level_function_definitions(symint_block)
                if not sig.is_member
            )
    return signatures


def signature_without_templates(signature: CppSignature) -> CppSignature:
    canonical = signature.canonical
    while canonical.startswith("template<"):
        close_pos = find_matching(canonical, canonical.find("<"), "<", ">")
        canonical = canonical[close_pos + 1 :]
    return CppSignature(
        raw=signature.raw,
        canonical=canonical,
        name=signature.name,
        is_member=signature.is_member,
    )


def find_tensor_class_body(text: str, class_name: str = "Tensor") -> str | None:
    clean = strip_comments_and_literals(text)
    match = re.search(
        rf"\bclass\s+(?:[A-Za-z_]\w*\s+)*{re.escape(class_name)}\b[^;{{]*\{{",
        clean,
    )
    if not match:
        return None
    open_pos = clean.find("{", match.start())
    close_pos = find_matching(clean, open_pos, "{", "}")
    return clean[open_pos + 1 : close_pos]


def public_class_sections(text: str, class_name: str = "Tensor") -> list[str]:
    body = find_tensor_class_body(text, class_name)
    if body is None:
        return []
    sections: list[str] = []
    access = "private"
    start = 0
    pos = 0
    angle = paren = bracket = brace = 0
    state = "normal"
    while pos < len(body):
        ch = body[pos]
        nxt = body[pos + 1] if pos + 1 < len(body) else ""
        if state == "normal":
            if ch == '"':
                state = "string"
                pos += 1
                continue
            if ch == "'":
                state = "char"
                pos += 1
                continue
        elif state in ("string", "char"):
            if ch == "\\" and nxt:
                pos += 2
                continue
            if (state == "string" and ch == '"') or (
                state == "char" and ch == "'"
            ):
                state = "normal"
            pos += 1
            continue

        if ch == "<":
            angle += 1
        elif ch == ">" and angle > 0:
            angle -= 1
        elif ch == "(":
            paren += 1
        elif ch == ")" and paren > 0:
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]" and bracket > 0:
            bracket -= 1
        elif ch == "{":
            brace += 1
        elif ch == "}" and brace > 0:
            brace -= 1

        if paren == 0 and bracket == 0 and brace == 0:
            match = re.match(r"\s*(public|protected|private)\s*:", body[pos:])
            if match:
                if access == "public":
                    sections.append(body[start:pos])
                access = match.group(1)
                pos += match.end()
                start = pos
                continue
        pos += 1
    if access == "public":
        sections.append(body[start:])
    return sections


def tensor_class_declarations(
    text: str,
    include_member_defaults: bool = False,
    class_name: str = "Tensor",
) -> list[CppSignature]:
    signatures: list[CppSignature] = []
    for body in public_class_sections(text, class_name):
        signatures.extend(
            class_section_declarations(
                body,
                include_member_defaults=include_member_defaults,
                class_name=class_name,
            )
        )
    return signatures


def class_section_declarations(
    body: str,
    include_member_defaults: bool = False,
    class_name: str = "Tensor",
) -> list[CppSignature]:
    signatures: list[CppSignature] = []
    start = 0
    pos = 0
    angle = paren = bracket = brace = 0
    state = "normal"
    while pos < len(body):
        ch = body[pos]
        nxt = body[pos + 1] if pos + 1 < len(body) else ""
        if state == "normal":
            if ch == '"':
                state = "string"
                pos += 1
                continue
            if ch == "'":
                state = "char"
                pos += 1
                continue
        elif state in ("string", "char"):
            if ch == "\\" and nxt:
                pos += 2
                continue
            if (state == "string" and ch == '"') or (
                state == "char" and ch == "'"
            ):
                state = "normal"
            pos += 1
            continue

        if ch == "<":
            angle += 1
        elif ch == ">" and angle > 0:
            angle -= 1
        elif ch == "(":
            paren += 1
        elif ch == ")" and paren > 0:
            paren -= 1
        elif ch == "[":
            bracket += 1
        elif ch == "]" and bracket > 0:
            bracket -= 1
        elif ch == "{":
            if angle == 0 and paren == 0 and bracket == 0 and brace == 0:
                candidate = body[start:pos]
                signature = parse_cpp_signature(
                    candidate,
                    class_member=True,
                    include_member_defaults=include_member_defaults,
                    class_name=class_name,
                )
                if signature is not None:
                    signatures.append(signature)
                end = find_matching(body, pos, "{", "}")
                pos = end + 1
                start = pos
                continue
            brace += 1
        elif ch == "}" and brace > 0:
            brace -= 1
        if ch == ";":
            if angle == 0 and paren == 0 and bracket == 0 and brace == 0:
                candidate = body[start:pos]
                signature = parse_cpp_signature(
                    candidate,
                    class_member=True,
                    include_member_defaults=include_member_defaults,
                    class_name=class_name,
                )
                if signature is not None:
                    signatures.append(signature)
                start = pos + 1
        pos += 1
    return signatures


def parse_torch_tensor_body(path: Path) -> list[CppSignature]:
    if not path.is_file():
        return []
    text = path.read_text()
    signatures: list[CppSignature] = []
    signatures.extend(
        tensor_class_declarations(text, include_member_defaults=True)
    )
    for block in namespace_at_blocks(text):
        signatures.extend(
            sig
            for sig in top_level_function_definitions(
                block, include_member_defaults=True
            )
            if sig.is_member
        )
    deduped: dict[str, CppSignature] = {}
    for sig in signatures:
        deduped.setdefault(sig.canonical, sig)
    return list(deduped.values())


def parse_torch_tensor_base(path: Path) -> list[CppSignature]:
    if not path.is_file():
        return []
    return tensor_class_declarations(
        path.read_text(), include_member_defaults=True, class_name="TensorBase"
    )


def parse_paddle_class_members(
    path: Path, class_name: str
) -> dict[str, CppSignature]:
    if not path.is_file():
        return {}
    text = path.read_text()
    declarations_with_defaults = tensor_class_declarations(
        text, include_member_defaults=True, class_name=class_name
    )
    declarations_without_defaults = tensor_class_declarations(
        text, include_member_defaults=False, class_name=class_name
    )
    declarations: dict[str, CppSignature] = {}
    for no_default, with_default in zip(
        declarations_without_defaults, declarations_with_defaults
    ):
        declarations.setdefault(no_default.canonical, with_default)
    return declarations


def parse_paddle_tensor_body(path: Path) -> dict[str, CppSignature]:
    return parse_paddle_class_members(path, "Tensor")


def parse_paddle_tensor_base(path: Path) -> dict[str, CppSignature]:
    return parse_paddle_class_members(path, "TensorBase")


def valid_torch_include_dir(path: Path) -> bool:
    return (path / "ATen/core/TensorBody.h").is_file() and (
        path / "ATen/ops"
    ).is_dir()


def torch_include_dirs_from_python() -> list[Path]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from torch.utils.cpp_extension import include_paths
    except Exception:
        return []
    return [Path(path) for path in include_paths()]


def discover_torch_include_dir(torch_include_dir: str | None) -> Path:
    candidates: list[Path] = []
    if torch_include_dir:
        candidates.append(Path(torch_include_dir))
    else:
        candidates.extend(torch_include_dirs_from_python())
    for candidate in candidates:
        if valid_torch_include_dir(candidate):
            return candidate
    searched = ", ".join(str(path) for path in candidates) or "<none>"
    raise FileNotFoundError(
        "Cannot find libtorch headers. Searched: "
        f"{searched}. Install CPU PyTorch with: {TORCH_INSTALL_HINT}"
    )


def git_ref_exists(paddle_root: Path, ref: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref],
        cwd=paddle_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def choose_base_ref(paddle_root: Path, branch: str) -> str:
    refs = [f"upstream/{branch}", f"origin/{branch}", branch]
    for ref in refs:
        if git_ref_exists(paddle_root, ref):
            return ref
    raise RuntimeError(
        "Cannot find a base ref for ATen ops signature check. Tried: "
        + ", ".join(refs)
    )


def added_aten_ops_headers(paddle_root: Path, branch: str) -> list[Path]:
    return changed_aten_ops_headers(paddle_root, branch)


def changed_paths(
    paddle_root: Path, branch: str, paths: Iterable[Path | str]
) -> list[Path]:
    base_ref = choose_base_ref(paddle_root, branch)
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=AM",
            base_ref,
            "--",
            *(str(path) for path in paths),
        ],
        cwd=paddle_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return [paddle_root / line.strip() for line in result.stdout.splitlines()]


def changed_aten_ops_headers(paddle_root: Path, branch: str) -> list[Path]:
    return [
        path
        for path in changed_paths(paddle_root, branch, [ATEN_OPS_DIR])
        if path.suffix == ".h"
    ]


def tensor_body_changed(paddle_root: Path, branch: str) -> bool:
    return any(
        path == paddle_root / ATEN_TENSOR_BODY
        for path in changed_paths(paddle_root, branch, [ATEN_TENSOR_BODY])
    )


def tensor_base_changed(paddle_root: Path, branch: str) -> bool:
    return any(
        path == paddle_root / ATEN_TENSOR_BASE
        for path in changed_paths(paddle_root, branch, [ATEN_TENSOR_BASE])
    )


def all_aten_ops_headers(paddle_root: Path) -> list[Path]:
    return sorted((paddle_root / ATEN_OPS_DIR).glob("*.h"))


def same_name_candidates(
    candidates: Iterable[CppSignature], signature: CppSignature
) -> tuple[CppSignature, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if candidate.name == signature.name
        and candidate.is_member == signature.is_member
    )


def check_header(
    paddle_header: Path,
    paddle_root: Path,
    torch_include_dir: Path,
    torch_member_signatures: list[CppSignature],
    paddle_member_signatures: dict[str, CppSignature],
) -> list[CheckError]:
    signatures, parse_errors = parse_paddle_header(paddle_header)
    errors = [
        CheckError(header=paddle_header, message=message)
        for message in parse_errors
    ]
    if parse_errors:
        return errors
    ops_name = paddle_header.name
    torch_ops_header = torch_include_dir / "ATen/ops" / ops_name
    torch_free_signatures = parse_torch_ops_header(torch_ops_header)
    for signature in signatures:
        if signature.is_member:
            torch_header = torch_include_dir / "ATen/core/TensorBody.h"
            candidates = torch_member_signatures
            comparison_signature = paddle_member_signatures.get(
                signature.canonical
            )
            if comparison_signature is None:
                errors.append(
                    CheckError(
                        header=paddle_header,
                        message=(
                            "member function implementation is missing a "
                            "matching Paddle TensorBody declaration"
                        ),
                        signature=signature,
                        torch_header=torch_header,
                        candidates=same_name_candidates(candidates, signature),
                    )
                )
                continue
        else:
            torch_header = torch_ops_header
            candidates = torch_free_signatures
            comparison_signature = signature
        if any(
            candidate.canonical == comparison_signature.canonical
            for candidate in candidates
        ):
            continue
        errors.append(
            CheckError(
                header=paddle_header,
                message="signature does not match libtorch",
                signature=comparison_signature,
                torch_header=torch_header,
                candidates=same_name_candidates(candidates, signature),
            )
        )
    return errors


def check_tensor_body_members(
    paddle_root: Path,
    torch_include_dir: Path,
    torch_member_signatures: list[CppSignature],
    paddle_member_signatures: dict[str, CppSignature],
) -> list[CheckError]:
    tensor_body = paddle_root / ATEN_TENSOR_BODY
    torch_header = torch_include_dir / "ATen/core/TensorBody.h"
    torch_canonicals = {
        signature.canonical for signature in torch_member_signatures
    }
    errors: list[CheckError] = []
    for signature in paddle_member_signatures.values():
        if is_paddle_internal_member(signature):
            continue
        if signature.canonical in torch_canonicals:
            continue
        errors.append(
            CheckError(
                header=tensor_body,
                message="TensorBody member signature does not match libtorch",
                signature=signature,
                torch_header=torch_header,
                candidates=same_name_candidates(
                    torch_member_signatures, signature
                ),
            )
        )
    return errors


def check_tensor_base_members(
    paddle_root: Path,
    torch_include_dir: Path,
    torch_base_signatures: list[CppSignature],
    paddle_base_signatures: dict[str, CppSignature],
) -> list[CheckError]:
    tensor_base = paddle_root / ATEN_TENSOR_BASE
    torch_header = torch_include_dir / "ATen/core/TensorBase.h"
    torch_canonicals = {
        signature.canonical for signature in torch_base_signatures
    }
    errors: list[CheckError] = []
    for signature in paddle_base_signatures.values():
        if is_paddle_internal_member(signature):
            continue
        if signature.canonical in torch_canonicals:
            continue
        errors.append(
            CheckError(
                header=tensor_base,
                message="TensorBase member signature does not match libtorch",
                signature=signature,
                torch_header=torch_header,
                candidates=same_name_candidates(
                    torch_base_signatures, signature
                ),
            )
        )
    return errors


def is_paddle_internal_member(signature: CppSignature) -> bool:
    return signature.name.startswith("_PD_") or signature.name.endswith("_impl")


def format_errors(errors: list[CheckError], paddle_root: Path) -> str:
    lines = ["ATen ops signature check failed:"]
    for idx, error in enumerate(errors, start=1):
        rel_header = error.header.relative_to(paddle_root)
        lines.append(f"{idx}. {rel_header}: {error.message}")
        if error.signature is None:
            continue
        lines.append(f"   Paddle signature: {error.signature.raw}")
        if error.torch_header is not None:
            lines.append(f"   Libtorch header: {error.torch_header}")
        if error.candidates:
            lines.append("   Libtorch candidates with the same name:")
            for candidate in error.candidates:
                lines.append(f"     - {candidate.raw}")
        else:
            lines.append("   Libtorch candidates with the same name: <none>")
    return "\n".join(lines)


def run_check(
    paddle_root: Path,
    torch_include_dir: Path,
    headers: list[Path],
    check_tensor_body: bool = True,
    check_tensor_base: bool = True,
) -> list[CheckError]:
    tensor_body = torch_include_dir / "ATen/core/TensorBody.h"
    torch_member_signatures = parse_torch_tensor_body(tensor_body)
    torch_base_signatures = parse_torch_tensor_base(
        torch_include_dir / "ATen/core/TensorBase.h"
    )
    paddle_tensor_body = paddle_root / ATEN_TENSOR_BODY
    paddle_member_signatures = parse_paddle_tensor_body(paddle_tensor_body)
    paddle_tensor_base = paddle_root / ATEN_TENSOR_BASE
    paddle_base_signatures = parse_paddle_tensor_base(paddle_tensor_base)
    errors: list[CheckError] = []
    if check_tensor_body:
        errors.extend(
            check_tensor_body_members(
                paddle_root=paddle_root,
                torch_include_dir=torch_include_dir,
                torch_member_signatures=torch_member_signatures,
                paddle_member_signatures=paddle_member_signatures,
            )
        )
    if check_tensor_base:
        errors.extend(
            check_tensor_base_members(
                paddle_root=paddle_root,
                torch_include_dir=torch_include_dir,
                torch_base_signatures=torch_base_signatures,
                paddle_base_signatures=paddle_base_signatures,
            )
        )
    for header in headers:
        errors.extend(
            check_header(
                paddle_header=header,
                paddle_root=paddle_root,
                torch_include_dir=torch_include_dir,
                torch_member_signatures=torch_member_signatures,
                paddle_member_signatures=paddle_member_signatures,
            )
        )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check changed Paddle compat ATen signatures against "
            "libtorch headers."
        )
    )
    parser.add_argument(
        "--paddle-root",
        default=".",
        help="Path to Paddle repository root.",
    )
    parser.add_argument(
        "--torch-include-dir",
        default=None,
        help="Path to libtorch include directory.",
    )
    parser.add_argument(
        "--branch",
        default=os.environ.get("BRANCH", "develop"),
        help="Base branch name used when collecting changed files.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Check all compat ATen ops headers, TensorBody members, and "
            "TensorBase members instead of only changed files."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paddle_root = Path(args.paddle_root).resolve()
    try:
        check_tensor_body = args.all or tensor_body_changed(
            paddle_root, args.branch
        )
        check_tensor_base = args.all or tensor_base_changed(
            paddle_root, args.branch
        )
        headers = (
            all_aten_ops_headers(paddle_root)
            if args.all or check_tensor_body
            else changed_aten_ops_headers(paddle_root, args.branch)
        )
        if not headers and not check_tensor_body and not check_tensor_base:
            print(
                "No changed compat ATen ops headers, TensorBody.h, or "
                "TensorBase.h found; skip check."
            )
            return 0
        torch_include_dir = discover_torch_include_dir(args.torch_include_dir)
        errors = run_check(
            paddle_root,
            torch_include_dir,
            headers,
            check_tensor_body=check_tensor_body,
            check_tensor_base=check_tensor_base,
        )
    except Exception as exc:
        print(f"ATen ops signature check error: {exc}", file=sys.stderr)
        return 1
    if errors:
        print(format_errors(errors, paddle_root), file=sys.stderr)
        return 1
    print(f"ATen ops signature check passed for {len(headers)} header(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
