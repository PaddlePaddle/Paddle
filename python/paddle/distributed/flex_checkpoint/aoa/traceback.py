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


class AOATraceback:
    """
    When error occurs, print the chain of "original aoa_statement -> ... -> current expression".
    """

    def __init__(self) -> None:
        self.records: list[dict] = []
        self.last_error_chain: list[str] = []
        self.last_error_message: str = ""
        self.last_error_stage: str = ""
        self.last_error_type: str = ""
        self.parent_map: dict[str, str | None] = {}
        self.child_macro_map: dict[str, str] = {}

    def register_roots(self, expressions: list[str]) -> None:
        """Register the original aoa_statements as the root nodes of the chain."""
        for expr in expressions:
            self.parent_map.setdefault(expr, None)

    def record_children(
        self, parent: str, children: list[str], macro_name: str | None = None
    ) -> None:
        """Record the children expressions obtained by the parent expression, and mark the macro name used."""
        macro = macro_name or "Expanded"
        for child in children:
            if child == parent:
                continue
            self.parent_map[child] = parent
            self.child_macro_map[child] = macro

    def build_chain(self, expr: str) -> list[str]:
        """Build the chain from the root to expr by tracing back from the current expression."""
        chain: list[str] = []
        visited = set()
        cur = expr
        while cur is not None and cur not in visited:
            chain.append(cur)
            visited.add(cur)
            cur = self.parent_map.get(cur)
        chain.reverse()
        return chain

    def add_error(
        self,
        error_message: str,
        stage: str,
        chain: list[str],
        error_type: str = "",
    ) -> None:
        """Record the error chain and information."""
        self.last_error_chain = chain
        self.last_error_message = error_message
        self.last_error_stage = stage
        self.last_error_type = error_type or ""
        self.records.append(
            {
                "type": "error",
                "stage": stage,
                "message": error_message,
                "error_type": self.last_error_type,
                "chain": chain,
            }
        )

    def format_traceback(self) -> str:
        lines: list[str] = []
        header_text = " AOA Traceback (related chain) "
        header = f"===={header_text}===="
        footer = "=" * len(header)

        if self.last_error_chain:
            lines.append(header)
            indent_unit = "    "

            lines.append("| Origin AOA Statement")
            origin_expr = self.last_error_chain[0].replace("\n", " ")
            lines.append(f"|-> {origin_expr}")

            for level, expr in enumerate(self.last_error_chain[1:], start=1):
                indent = indent_unit * level
                single_line_expr = expr.replace("\n", " ")
                macro = self.child_macro_map.get(
                    expr, self.last_error_stage or "Expanded"
                )
                lines.append(f"{indent}| {macro}")
                lines.append(f"{indent}|-> {single_line_expr}")

            if self.last_error_message:
                err_title = self.last_error_type or "Error"
                stage_str = (
                    f" [{self.last_error_stage}]"
                    if self.last_error_stage
                    else ""
                )
                err_level = len(self.last_error_chain)
                indent = indent_unit * err_level
                single_line_msg = self.last_error_message.replace("\n", " ")
                lines.append(f"{indent}| Error")
                lines.append(
                    f"{indent}|-> ({err_title}{stage_str}) {single_line_msg}"
                )

            lines.append(footer)
        else:
            lines.append(header)
            lines.append("(No trace records)")
            lines.append(footer)

        return "\n".join(lines)

    def print(self, logger=None) -> None:
        text = self.format_traceback()
        if logger:
            logger.error(text)
        else:
            print(text)
