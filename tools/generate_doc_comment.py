# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import paddle  # noqa: F401


def load_api_by_name(path: str) -> Callable[..., Any] | None:
    """
    Recursively resolves a string path to a Python object.
    """
    if not path:
        return None

    # First, try to import the entire path as a module (e.g., "paddle" or "paddle.autograd").
    try:
        return importlib.import_module(path)
    except ImportError:
        # If the import fails, it might be an object within a module.
        # If there's no dot, it was a failed top-level import, so we can't proceed.
        if "." not in path:
            return None

        # Split the path into its parent and the final object name.
        # e.g., "paddle.Tensor" -> parent="paddle", child="Tensor"
        parent_path, child_name = path.rsplit('.', 1)
        parent_obj = load_api_by_name(parent_path)

        # If the parent object could not be resolved, we can't find the child.
        if parent_obj is None:
            return None

        # Use getattr with a default value to safely get the child object.
        return getattr(parent_obj, child_name, None)


def generate_comment_body(doc_diff: str, pr_id: int) -> str:
    if not doc_diff:
        return ""

    output_lines: list[str] = []
    base_url = f"http://preview-paddle-pr-{pr_id}.paddle-docs-preview.paddlepaddle.org.cn/documentation/docs/en/api"

    # Extract API names like 'paddle.autograd.backward' from lines like:
    # - paddle.autograd.backward (ArgSpec(...), ('document', ...))
    # + paddle.autograd.backward (ArgSpec(...), ('document', ...))
    apis: list[str] = sorted(
        set(re.findall(r"^[+]\s*([a-zA-Z0-9_.]+)\s*\(", doc_diff, re.MULTILINE))
    )
    # All apis should be loaded, this seems a explicitly check.
    unload_apis: list[str] = []

    if not apis:
        return ""

    for api in apis:
        api_obj = load_api_by_name(api)

        if api_obj is None:
            unload_apis.append(api)
            continue

        api_path = api.replace('.', '/')
        url = f"{base_url}/{api_path}_en.html"

        if "." in api:
            parent_path, child_name = api.rsplit('.', 1)
            parent_obj = load_api_by_name(parent_path)
            if inspect.isclass(parent_obj) and not inspect.isclass(api_obj):
                parent_api_path = parent_path.replace('.', '/')
                url = f"{base_url}/{parent_api_path}_en.html#{child_name}"

        output_lines.append(f"- **{api}**: [Preview]({url})")
    unload_error_msg = (
        f"@ooooo-create, following apis cannot be loaded, please check it: {', '.join(unload_apis)}"
        if unload_apis
        else ""
    )

    if not output_lines:
        return unload_error_msg

    api_links = "\n".join(output_lines)
    comment_body = f"""<details>
<summary>📚 Preview documentation links for API changes in this PR (Click to expand)</summary>

{unload_error_msg}

<table>
<tr>
<td>
ℹ️ <b>Preview Notice</b><br>
Please wait for the <code>Doc-Preview</code> workflow to complete before clicking the preview links below, otherwise you may see outdated content.
</td>
</tr>
</table>

The following are preview links for new or modified API documentation in this PR:

{api_links}

</details>"""

    return comment_body


def cli():
    parser = argparse.ArgumentParser(
        description="Generate documentation comment for PR with API changes"
    )
    parser.add_argument(
        "doc_diff_path", help="Path to the documentation diff file", type=str
    )
    parser.add_argument("pr_id", help="Pull request ID", type=int)
    return parser.parse_args()


def main():
    args = cli()

    with open(args.doc_diff_path, 'r') as f:
        doc_diff_content = f.read()

    comment = generate_comment_body(doc_diff_content, args.pr_id)
    print(comment)


if __name__ == "__main__":
    main()
