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

import atexit
import base64
import json
import sys
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

from typing_extensions import Self

from .envs import ENV_SOT_COLLECT_INFO, ENV_SOT_SERIALIZE_INFO
from .utils import Singleton

if TYPE_CHECKING:
    import types

    from .exceptions import BreakGraphReasonBase

PREFIX = "<sot>"
SUFFIX = "</sot>"
ENCODING = "utf-8"


def try_import_graphviz():
    try:
        import graphviz

        return graphviz
    except ImportError:
        return None


class InfoType(Enum):
    STEP_INFO = 0
    E2E_INFO = 1


class InfoCollector(metaclass=Singleton):
    def __init__(self):
        self._step_info: dict[str, list[InfoBase]] = {}
        self._e2e_info: dict[str, list[InfoBase]] = {}

    def get_info_dict(self, info_type: InfoType) -> dict[str, list[InfoBase]]:
        if info_type == InfoType.STEP_INFO:
            return self._step_info
        else:
            return self._e2e_info

    def attach(self, cls: type[InfoBase], *args, **kwargs) -> None:
        if self.need_collect(cls):
            info = cls(*args, **kwargs)
            self.register(info)

    def register(self, info: InfoBase) -> None:
        info_class_name = info.__class__.__name__
        info_type = info.TYPE
        info_dict = self.get_info_dict(info_type)
        info_dict.setdefault(info_class_name, [])
        info_dict[info_class_name].append(info)

    def need_collect(self, cls: type[InfoBase]) -> bool:
        return cls.SHORT_NAME in ENV_SOT_COLLECT_INFO.get()

    def clear_step_info(self):
        self._step_info.clear()

    def clear_e2e_info(self):
        self._e2e_info.clear()

    def clear(self):
        self.clear_step_info()
        self.clear_e2e_info()

    def print_step_report(self):
        self.print_report(InfoType.STEP_INFO)

    def print_e2e_info_atexit(self) -> None:
        def atexit_hook():
            self.print_report(InfoType.E2E_INFO)
            sys.stdout.flush()
            self.clear()

        atexit.register(atexit_hook)

    def print_report(self, info_type: InfoType) -> None:
        if info_dict := self.get_info_dict(info_type):
            print(self.generate_report(info_dict))

    def generate_report(self, info_dict: dict[str, list[InfoBase]]) -> str:
        report = ""
        for info_class_name, info_list in info_dict.items():
            cls = info_list[0].__class__
            report += f"{info_class_name} ({cls.SHORT_NAME}):\n"
            if ENV_SOT_SERIALIZE_INFO.get():
                report += cls.json_report(info_list)
            else:
                report += cls.summary(info_list)
            report += "\n"
        return report


InfoCollector().print_e2e_info_atexit()


class InfoBase(ABC):
    SHORT_NAME: ClassVar[str]
    TYPE: ClassVar[InfoType]

    def __init__(self): ...

    @classmethod
    @abstractmethod
    def summary(cls, history: list[Self]) -> str: ...

    @classmethod
    def serialize(cls, obj: dict[str:Any]) -> str:

        json_data = json.dumps(obj)
        b64_bytes = base64.b64encode(json_data.encode(ENCODING))

        return b64_bytes.decode(ENCODING)

    @classmethod
    def deserialize(cls, data: bytes | str) -> dict:
        if isinstance(data, str):
            data = data.encode(ENCODING)
        json_str = base64.b64decode(data).decode(ENCODING)

        return json.loads(json_str)


class NewSymbolHitRateInfo(InfoBase):
    SHORT_NAME = "new_symbol_hit_rate"
    TYPE = InfoType.STEP_INFO

    def __init__(
        self, input_tensor_ids: list[int], output_tensor_ids: list[int]
    ):
        super().__init__()
        self.input_tensor_ids = input_tensor_ids
        self.output_tensor_ids = output_tensor_ids

    @classmethod
    def summary(cls, history: list[Self]) -> str:
        if len(history) == 0:
            return f"No {cls.SHORT_NAME} info"
        if len(history) == 1:
            return "Only one subgraph is generated"
        known_tensor_ids = set()
        hit_count = 0
        all_count = sum([len(info.input_tensor_ids) for info in history[1:]])
        for i, info in enumerate(history):
            for tensor_id in info.input_tensor_ids:
                # Skip the first graph
                if i == 0:
                    continue
                if tensor_id in known_tensor_ids:
                    hit_count += 1
            for tensor_id in info.output_tensor_ids:
                known_tensor_ids.add(tensor_id)
        summary = f"All tensor count: {all_count}, hit count: {hit_count}\n"
        summary += f"Hit rate: {hit_count / all_count:.2f}"
        return summary

    @classmethod
    def json_report(cls, history: list[Self]) -> str:
        # TODO: need to support serialize the output
        return cls.summary(history)


class SubGraphRelationInfo(InfoBase):
    SHORT_NAME = "subgraph_relation"
    TYPE = InfoType.STEP_INFO
    STEP_UNIQUE_ID = 0

    class ConcreteShapeInfo(NamedTuple):
        id: int
        ir_shape: list[int]
        real_shape: list[int]

    def __init__(
        self,
        subgraph_name: str,
        input_shape_infos: list[SubGraphRelationInfo.ConcreteShapeInfo],
        output_shape_infos: list[SubGraphRelationInfo.ConcreteShapeInfo],
        is_first_call: bool,
        graph_size: int,
    ):
        super().__init__()
        self.subgraph_name = subgraph_name
        self.input_shape_infos = input_shape_infos
        self.output_shape_infos = output_shape_infos
        self.is_first_call = is_first_call
        self.graph_size = graph_size

    @classmethod
    def summary(cls, history: list[Self]) -> str:
        # TODO: attach input shape (with dynamic shape info)
        cls.STEP_UNIQUE_ID += 1
        if len(history) == 0:
            return f"No {cls.SHORT_NAME} info"
        if all(not subgraph_info.is_first_call for subgraph_info in history):
            return "All subgraph are not the first call"
        graphviz = try_import_graphviz()
        if graphviz is None:
            return "Please install graphviz to show the subgraph relation"
        dot = graphviz.Digraph()
        shape_infos = [
            shape_info
            for info in history
            for shape_info in info.input_shape_infos + info.output_shape_infos
        ]

        def to_tensor_node_name(
            shape_info: SubGraphRelationInfo.ConcreteShapeInfo,
        ):
            return f"tensor_{shape_info.id}"

        visited_shape = set()
        for shape_info in shape_infos:
            if shape_info.id in visited_shape:
                continue
            visited_shape.add(shape_info.id)
            dot.node(
                to_tensor_node_name(shape_info),
                f"Tensor {shape_info.id} shape={shape_info.real_shape}",
                shape="rect",
            )
        for i, info in enumerate(history):
            subgraph_id = f"subgraph_{i}"
            dot.node(
                subgraph_id,
                f"Subgraph {i} ({info.subgraph_name}, size={info.graph_size})",
                shape="oval",
                fillcolor="cyan" if info.is_first_call else None,
                style="filled" if info.is_first_call else None,
            )
            for shape_info in info.input_shape_infos:
                dot.edge(
                    to_tensor_node_name(shape_info),
                    subgraph_id,
                    label=str(shape_info.ir_shape),
                )
            for shape_info in info.output_shape_infos:
                dot.edge(
                    subgraph_id,
                    to_tensor_node_name(shape_info),
                    label=str(shape_info.ir_shape),
                )

        directory = Path(".") / "subgraph_relation"
        directory.mkdir(exist_ok=True, parents=True)
        filename = f"subgraph_relation_{cls.STEP_UNIQUE_ID}"
        dot.render(directory / filename, format="svg", cleanup=True)
        return f"Please check {directory / filename}.svg for subgraph relation"

    @classmethod
    def json_report(cls, history: list[Self]) -> str:
        # TODO: need to support serialize the output
        return cls.summary(history)


class CompileCountInfo(InfoBase):
    SHORT_NAME = "compile_count"
    TYPE = InfoType.E2E_INFO

    def __init__(self, code: types.CodeType):
        super().__init__()
        self.code = code

    @classmethod
    def summary(cls, history: list[Self]) -> str:
        if len(history) == 0:
            return f"No {cls.SHORT_NAME} info"
        code_count = {}
        for info in history:
            code_count[info.code] = code_count.get(info.code, 0) + 1
        summary_lines = []
        for code, count in sorted(
            code_count.items(), key=lambda x: x[1], reverse=True
        ):
            filename, lineno = code.co_filename, code.co_firstlineno
            summary_lines.append(
                f"    {code.co_name} ({filename}:{lineno}): {count}"
            )
        summary = "\n".join(summary_lines)
        return summary

    @classmethod
    def json_report(cls, history: list[Self]) -> str:
        # TODO: need to support serialize the output
        return cls.summary(history)


class BreakGraphReasonInfo(InfoBase):
    SHORT_NAME = "breakgraph_reason"
    TYPE = InfoType.E2E_INFO

    def __init__(self, reason: BreakGraphReasonBase):
        super().__init__()
        self.reason = reason

    @classmethod
    def classify(cls, history: list[Self]) -> str:
        reasons_dict = {}

        for info in history:
            name = info.reason.__class__.__name__
            if name not in reasons_dict:
                reasons_dict[name] = []
            reasons_dict[name].append(str(info.reason))

        sorted_reasons = list(reasons_dict.items())
        sorted_reasons.sort(key=lambda x: len(x[1]), reverse=True)

        return reasons_dict, sorted_reasons

    @classmethod
    def summary(cls, history: list[Self]) -> str:

        reason_dict, reason_list = cls.classify(history)

        return "\n".join(
            [
                f"{name} ({len(reasons)}):\n\t" + "\n\t".join(reasons)
                for name, reasons in reason_list
            ]
        )

    @classmethod
    def json_report(cls, history: list[Self]) -> str:

        reason_dict, sorted_reasons = cls.classify(history)
        reason_dict["count"] = {k: len(v) for k, v in sorted_reasons}
        serialized = cls.serialize({cls.SHORT_NAME: reason_dict})

        return f"{PREFIX}{serialized}{SUFFIX}"

    @classmethod
    def restore_from_string(cls, serialized: str) -> list[Self]:
        # This method is the inverse of json_report

        from paddle.jit.sot.utils import exceptions

        history = []
        obj = cls.deserialize(serialized)[cls.SHORT_NAME]
        obj.pop("count")

        for classname in obj:

            ReasonClass = getattr(exceptions, classname, None)
            for reason in obj[classname]:
                history.append(cls(ReasonClass(reason_str=reason)))

        return history

    @staticmethod
    def collect_break_graph_reason(reason: BreakGraphReasonBase):
        if not InfoCollector().need_collect(BreakGraphReasonInfo):
            return

        InfoCollector().attach(BreakGraphReasonInfo, reason)


class SubGraphInfo(InfoBase):
    SHORT_NAME = "subgraph_info"
    TYPE = InfoType.STEP_INFO

    def __init__(self, graph: str, op_num: int, sir_name: str):
        # NOTE: All data should be serializable
        super().__init__()
        self.graph = graph
        self.op_num = op_num
        self.sir_name = sir_name

    def __str__(self):
        return f"[SIR Name]: {self.sir_name}   [OpNum]: {self.op_num}\n{self.graph}"

    @classmethod
    def summary(cls, history: list[Self]) -> str:
        num_of_subgraph = len(history)
        sum_of_op_num = sum(item.op_num for item in history)

        need_details = "details" in ENV_SOT_COLLECT_INFO.get().get(
            cls.SHORT_NAME, []
        )

        details = ""
        if need_details:
            details = "\n".join(
                [
                    f"[SubGraphIdx]: {idx}   {info}"
                    for idx, info in enumerate(map(str, history))
                ]
            )

        summary = f"[Number of subgraph]: {num_of_subgraph} [Sum of opnum]: {sum_of_op_num}"

        return f"{summary}\n{details}"

    @classmethod
    def json_report(cls, history: list[Self]) -> str:
        need_details = "details" in ENV_SOT_COLLECT_INFO.get().get(
            cls.SHORT_NAME, []
        )

        aggregated_info_list = []
        for idx, record in enumerate(history):
            entry_data = {}

            entry_data["SIR_name"] = record.sir_name
            entry_data["OpNum"] = record.op_num
            entry_data["Graph"] = ""
            if need_details:
                entry_data["Graph"] = str(record.graph)
            aggregated_info_list.append(entry_data)

        serialized = cls.serialize({cls.SHORT_NAME: aggregated_info_list})

        return f"{PREFIX}{serialized}{SUFFIX}"

    @classmethod
    def restore_from_string(cls, serialized: str) -> list[Self]:
        # This method is the inverse of json_report

        history = []
        obj = cls.deserialize(serialized)[cls.SHORT_NAME]

        for entry in obj:

            history.append(
                SubGraphInfo(
                    graph=entry["Graph"],
                    op_num=entry["OpNum"],
                    sir_name=entry["SIR_name"],
                )
            )

        return history

    def __eq__(self, other):

        need_graph_equal = "details" in ENV_SOT_COLLECT_INFO.get().get(
            self.SHORT_NAME, []
        )

        graph_equal_or_not = True
        if need_graph_equal:
            graph_equal_or_not = self.graph == other.graph

        return (
            graph_equal_or_not
            and self.op_num == other.op_num
            and self.sir_name == other.sir_name
        )
