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

"""Check Linux wheel ABI compatibility by comparing protected ELF symbols.

The check is intentionally one-way: symbols added by a PR are allowed, while
protected symbols present in the base wheel must still exist in the PR wheel.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

WHEEL_LIBRARY_PATHS = (
    "paddle/base/libpaddle.so",
    "paddle/libs/libphi.so",
    "paddle/libs/libphi_core.so",
    "paddle/libs/libphi_gpu.so",
)

DEFINED_DYNAMIC_SYMBOL_TYPES = {"FUNC", "OBJECT"}

PROTECTED_COMPAT_CXX_PREFIXES = (
    "c10::",
    "at::",
    "torch::",
    "caffe2::",
)

PROTECTED_COMPAT_MANGLED_CXX_PREFIXES = (
    "_ZN2at",
    "_ZNK2at",
    "_ZN3c10",
    "_ZNK3c10",
    "_ZN5torch",
    "_ZNK5torch",
    "_ZN6caffe2",
    "_ZNK6caffe2",
)

REQUIRED_ABI_APPROVERS = ("SigureMo", "BingooYang")
DEFAULT_GITHUB_REPOSITORY = "PaddlePaddle/Paddle"
GITHUB_API_URL = "https://api.github.com"


@dataclass(frozen=True)
class DynamicSymbol:
    name: str
    symbol_type: str
    bind: str
    section: str
    demangled_name: str


@dataclass(frozen=True)
class RemovedSymbol:
    library: str
    name: str
    demangled_name: str


@dataclass(frozen=True)
class MissingLibrary:
    library: str


@dataclass(frozen=True)
class ApprovalCheckResult:
    approved: bool
    reviewer: str | None = None
    reason: str | None = None


def strip_elf_symbol_version(symbol_name: str) -> str:
    if "@@" in symbol_name:
        return symbol_name.split("@@", 1)[0]
    if "@" in symbol_name:
        return symbol_name.split("@", 1)[0]
    return symbol_name


def parse_readelf_dynamic_symbols(readelf_output: str) -> list[DynamicSymbol]:
    symbols = []
    for line in readelf_output.splitlines():
        fields = line.split()
        if len(fields) < 8 or not fields[0].endswith(":"):
            continue
        symbol_type = fields[3]
        bind = fields[4]
        section = fields[6]
        name = fields[7]
        if (
            bind != "GLOBAL"
            or section == "UND"
            or symbol_type not in DEFINED_DYNAMIC_SYMBOL_TYPES
        ):
            continue
        symbols.append(
            DynamicSymbol(
                name=name,
                symbol_type=symbol_type,
                bind=bind,
                section=section,
                demangled_name=strip_elf_symbol_version(name),
            )
        )
    return symbols


def demangle_symbol_names(symbol_names: Iterable[str]) -> dict[str, str]:
    unique_names = sorted(
        {strip_elf_symbol_version(name) for name in symbol_names}
    )
    if not unique_names:
        return {}
    cxxfilt = shutil.which("c++filt")
    if cxxfilt is None:
        return {name: name for name in unique_names}

    try:
        result = subprocess.run(
            [cxxfilt],
            input="\n".join(unique_names),
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {name: name for name in unique_names}

    demangled = result.stdout.splitlines()
    if len(demangled) != len(unique_names):
        return {name: name for name in unique_names}
    return dict(zip(unique_names, demangled))


def attach_demangled_names(
    symbols: Iterable[DynamicSymbol],
) -> list[DynamicSymbol]:
    symbol_list = list(symbols)
    demangled_names = demangle_symbol_names(
        symbol.name for symbol in symbol_list
    )
    return [
        DynamicSymbol(
            name=symbol.name,
            symbol_type=symbol.symbol_type,
            bind=symbol.bind,
            section=symbol.section,
            demangled_name=demangled_names.get(
                strip_elf_symbol_version(symbol.name), symbol.demangled_name
            ),
        )
        for symbol in symbol_list
    ]


def is_protected_paddle_abi_symbol(symbol: DynamicSymbol) -> bool:
    demangled = symbol.demangled_name
    if demangled.startswith(PROTECTED_COMPAT_CXX_PREFIXES):
        return True

    raw_name = strip_elf_symbol_version(symbol.name)
    return raw_name.startswith(PROTECTED_COMPAT_MANGLED_CXX_PREFIXES)


def protected_symbols_by_name(
    symbols: Iterable[DynamicSymbol],
) -> dict[str, DynamicSymbol]:
    return {
        symbol.name: symbol
        for symbol in symbols
        if is_protected_paddle_abi_symbol(symbol)
    }


def read_dynamic_symbols(library_path: str) -> list[DynamicSymbol]:
    try:
        result = subprocess.run(
            ["readelf", "--dyn-syms", "-W", library_path],
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("readelf is required to check ABI symbols") from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to read dynamic symbols from {library_path}:\n{exc.stderr}"
        ) from exc

    return attach_demangled_names(parse_readelf_dynamic_symbols(result.stdout))


def extract_wheel_libraries(
    wheel_path: str, library_paths: Iterable[str], output_dir: str
) -> dict[str, str]:
    extracted_libraries = {}
    with zipfile.ZipFile(wheel_path) as wheel:
        wheel_entries = set(wheel.namelist())
        for library_path in library_paths:
            if library_path not in wheel_entries:
                continue
            extracted_path = wheel.extract(library_path, output_dir)
            extracted_libraries[library_path] = extracted_path
    return extracted_libraries


def compare_library_symbols(
    library: str,
    base_symbols: Iterable[DynamicSymbol] | None,
    pr_symbols: Iterable[DynamicSymbol] | None,
) -> list[RemovedSymbol | MissingLibrary]:
    if base_symbols is None:
        return []
    if pr_symbols is None:
        return [MissingLibrary(library=library)]

    base_protected_symbols = protected_symbols_by_name(base_symbols)
    pr_protected_symbols = protected_symbols_by_name(pr_symbols)
    removed_names = sorted(
        set(base_protected_symbols) - set(pr_protected_symbols)
    )
    return [
        RemovedSymbol(
            library=library,
            name=name,
            demangled_name=base_protected_symbols[name].demangled_name,
        )
        for name in removed_names
    ]


def resolve_wheel_path(pattern: str, label: str) -> str:
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {label} wheel matching {pattern}, "
            f"but found {len(matches)}: {matches}"
        )
    return matches[0]


def compare_wheel_abi(
    base_wheel: str, pr_wheel: str, library_paths: Iterable[str]
) -> list[RemovedSymbol | MissingLibrary]:
    with tempfile.TemporaryDirectory(prefix="paddle_abi_check_") as temp_dir:
        base_dir = os.path.join(temp_dir, "base")
        pr_dir = os.path.join(temp_dir, "pr")
        base_libraries = extract_wheel_libraries(
            base_wheel, library_paths, base_dir
        )
        pr_libraries = extract_wheel_libraries(pr_wheel, library_paths, pr_dir)

        issues: list[RemovedSymbol | MissingLibrary] = []
        for library in library_paths:
            base_path = base_libraries.get(library)
            pr_path = pr_libraries.get(library)
            base_symbols = (
                read_dynamic_symbols(base_path)
                if base_path is not None
                else None
            )
            pr_symbols = (
                read_dynamic_symbols(pr_path) if pr_path is not None else None
            )
            issues.extend(
                compare_library_symbols(library, base_symbols, pr_symbols)
            )
        return issues


def format_issues(
    issues: Iterable[RemovedSymbol | MissingLibrary],
    max_report: int,
    title: str = "ABI compatibility check failed.",
) -> str:
    issue_list = list(issues)
    lines = [
        title,
        "The PR wheel removed protected dynamic symbols that exist in the base "
        "wheel. Removing these symbols can break downstream wheels or shared "
        "libraries compiled against the base branch.",
        "",
    ]
    for issue in issue_list[:max_report]:
        if isinstance(issue, MissingLibrary):
            lines.extend(
                [
                    f"Library: {issue.library}",
                    "  PR wheel is missing this library, but the base wheel "
                    "contains it.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    f"Library: {issue.library}",
                    f"  Raw symbol: {issue.name}",
                    f"  Demangled: {issue.demangled_name}",
                    "",
                ]
            )

    omitted_count = len(issue_list) - max_report
    if omitted_count > 0:
        lines.append(f"... omitted {omitted_count} additional removed symbols.")
    return "\n".join(lines)


def find_required_abi_approver(
    reviews: Iterable[dict],
    required_approvers: Iterable[str] = REQUIRED_ABI_APPROVERS,
) -> str | None:
    required_logins = {login.lower() for login in required_approvers}
    for review in reviews:
        if review.get("state") != "APPROVED":
            continue
        user = review.get("user")
        if not isinstance(user, dict):
            continue
        login = user.get("login")
        if isinstance(login, str) and login.lower() in required_logins:
            return login
    return None


def fetch_pr_reviews(
    pr_id: str,
    token: str,
    repository: str = DEFAULT_GITHUB_REPOSITORY,
    api_url: str = GITHUB_API_URL,
) -> list[dict]:
    reviews: list[dict] = []
    page = 1
    while True:
        url = (
            f"{api_url}/repos/{repository}/pulls/{pr_id}/reviews"
            f"?per_page=100&page={page}"
        )
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"token {token}",
                "User-Agent": "Paddle-ABI-Compatibility-Check",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                page_reviews = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"failed to fetch PR reviews from GitHub: HTTP {exc.code}"
            ) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"failed to fetch PR reviews from GitHub: {exc}"
            ) from exc

        if not isinstance(page_reviews, list):
            raise RuntimeError("GitHub PR reviews response is not a list")

        reviews.extend(page_reviews)
        if len(page_reviews) < 100:
            break
        page += 1
    return reviews


def check_abi_removal_approval(
    env: dict[str, str] | None = None,
    fetch_reviews=fetch_pr_reviews,
) -> ApprovalCheckResult:
    env = os.environ if env is None else env
    pr_id = env.get("GIT_PR_ID") or env.get("PR_ID")
    token = (
        env.get("GITHUB_API_TOKEN")
        or env.get("GITHUB_TOKEN")
        or env.get("GH_TOKEN")
    )
    repository = env.get("GITHUB_REPOSITORY", DEFAULT_GITHUB_REPOSITORY)

    if not pr_id:
        return ApprovalCheckResult(
            approved=False, reason="GIT_PR_ID or PR_ID is not set"
        )
    if not token:
        return ApprovalCheckResult(
            approved=False,
            reason="GITHUB_API_TOKEN, GITHUB_TOKEN, or GH_TOKEN is not set",
        )

    try:
        reviews = fetch_reviews(pr_id, token, repository)
    except RuntimeError as exc:
        return ApprovalCheckResult(approved=False, reason=str(exc))

    reviewer = find_required_abi_approver(reviews)
    if reviewer is not None:
        return ApprovalCheckResult(approved=True, reviewer=reviewer)
    return ApprovalCheckResult(
        approved=False,
        reason=(
            "no APPROVED review from "
            + " or ".join(REQUIRED_ABI_APPROVERS)
            + " was found"
        ),
    )


def check_abi_issues_approval(
    issues: Iterable[RemovedSymbol | MissingLibrary],
    env: dict[str, str] | None = None,
    fetch_reviews=fetch_pr_reviews,
) -> ApprovalCheckResult:
    if not list(issues):
        return ApprovalCheckResult(approved=True, reason="no ABI issues")
    return check_abi_removal_approval(env=env, fetch_reviews=fetch_reviews)


def format_approval_failure(approval: ApprovalCheckResult) -> str:
    lines = [
        "You must have one RD (SigureMo or BingooYang) approval for protected "
        "ABI symbol removals.",
    ]
    if approval.reason:
        lines.append(f"Approval check failed: {approval.reason}.")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    paddle_root = os.environ.get("PADDLE_ROOT", os.getcwd())
    parser = argparse.ArgumentParser(
        description="Check Linux wheel ABI compatibility for Paddle symbols."
    )
    parser.add_argument(
        "--base-wheel",
        default=os.path.join(paddle_root, "build/dev_whl/*.whl"),
        help="Base branch wheel path or glob pattern.",
    )
    parser.add_argument(
        "--pr-wheel",
        default=os.path.join(paddle_root, "build/pr_whl/*.whl"),
        help="PR wheel path or glob pattern.",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=200,
        help="Maximum number of ABI issues to print.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        base_wheel = resolve_wheel_path(args.base_wheel, "base")
        pr_wheel = resolve_wheel_path(args.pr_wheel, "PR")
        issues = compare_wheel_abi(base_wheel, pr_wheel, WHEEL_LIBRARY_PATHS)
    except RuntimeError as exc:
        print(f"ABI compatibility check failed: {exc}", file=sys.stderr)
        return 1

    if issues:
        approval = check_abi_issues_approval(issues)
        if approval.approved:
            reviewer = approval.reviewer or "a required reviewer"
            print(
                format_issues(
                    issues,
                    args.max_report,
                    title="ABI compatibility issues found.",
                ),
                file=sys.stderr,
            )
            print(
                f"Protected ABI symbol removals were approved by {reviewer}; "
                "continuing.",
                file=sys.stderr,
            )
            return 0
        print(format_issues(issues, args.max_report), file=sys.stderr)
        print(format_approval_failure(approval), file=sys.stderr)
        return 1

    print("ABI compatibility check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
