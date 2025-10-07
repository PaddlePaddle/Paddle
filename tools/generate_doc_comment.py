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

import importlib
import inspect
import re
import sys

import paddle  # noqa: F401


def resolve_string_to_obj(path: str):
    """
    Recursively resolves a string path to a Python object.
    Handles modules, functions, classes, and methods.
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
        parent_obj = resolve_string_to_obj(parent_path)

        # If the parent object could not be resolved, we can't find the child.
        if parent_obj is None:
            return None

        # Use getattr with a default value to safely get the child object.
        return getattr(parent_obj, child_name, None)


def generate_comment_body(doc_diff, pr_id):
    if not doc_diff:
        return ""

    output_lines = []
    base_url = f"http://preview-paddle-pr-{pr_id}.paddle-docs-preview.paddlepaddle.org.cn/documentation/docs/en/api"

    # Extract API names like 'paddle.autograd.backward' from lines like:
    # - paddle.autograd.backward (ArgSpec(...), ('document', ...))
    # + paddle.autograd.backward (ArgSpec(...), ('document', ...))
    apis = sorted(
        set(re.findall(r"^[+]\s*([a-zA-Z0-9_.]+)\s*\(", doc_diff, re.MULTILINE))
    )

    for api in apis:
        api_obj = resolve_string_to_obj(api)

        if api_obj is None:
            raise ValueError(f"Could not resolve API path: {api}")

        api_path = api.replace('.', '/')
        url = f"{base_url}/{api_path}_en.html"

        if "." in api:
            parent_path, child_name = api.rsplit('.', 1)
            parent_obj = resolve_string_to_obj(parent_path)
            if inspect.isclass(parent_obj) and inspect.isfunction(api_obj):
                parent_api_path = parent_path.replace('.', '/')
                url = f"{base_url}/{parent_api_path}_en.html#{child_name}"

        output_lines.append(f"- [{api}]({url})")

    if not output_lines:
        return ""

    comment_body = """<details>
<summary>📚 因为涉及修改 api docstring，生成本次 PR 文档预览链接 (点击展开)</summary>

以下是本次 PR 中新增或变更文档的预览链接：

{}

</details>""".format("\n".join(output_lines))

    return comment_body


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python generate_doc_comment.py <path_to_doc_diff> <pr_id>"
        )
        sys.exit(1)

    doc_diff_path = sys.argv[1]
    pr_id = sys.argv[2]

    with open(doc_diff_path, 'r') as f:
        doc_diff_content = f.read()

    comment = generate_comment_body(doc_diff_content, pr_id)
    print(comment)
