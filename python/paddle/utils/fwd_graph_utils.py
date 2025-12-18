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

import io
import os
import re
import sys
import tempfile
from contextlib import contextmanager

import paddle
from paddle.base.wrapped_decorator import signature_safe_contextmanager


def _parse_tensors(input_str):
    input_tuples = re.findall(
        r"\(\s*(\w+)\s*,\s*(\{.*?\}|\[.*?\])\s*\)", input_str, re.DOTALL
    )
    tensors_list = []
    for var_name, value_str in input_tuples:
        # Only one Tensor
        if value_str.startswith('{'):
            tensor_info = _parse_tensor_info(value_str)
            tensors_list.append((var_name, [tensor_info]))
        # Tensor list
        elif value_str.startswith('['):
            list_content = value_str.strip('[]')
            dict_strs = _split_list_elements(list_content)
            tensor_list = []
            for dict_str in dict_strs:
                tensor_info = _parse_tensor_info(dict_str)
                if tensor_info:
                    tensor_list.append(tensor_info)
            tensors_list.append((var_name, tensor_list))
    return tensors_list


def _parse_api_name(debug_str):
    api_name_match = re.search(r"API_Name:\s*(\w+)", debug_str)
    return api_name_match.group(1) if api_name_match else None


def _parse_input_tensors(debug_str):
    input_match = re.search(r"Input:\s*(\[.*?\]  )", debug_str, re.DOTALL)
    return _parse_tensors(input_match.group(1)) if input_match else []


def _parse_output_tensors(debug_str):
    output_match = re.search(r"Output:\s*(\[.*?\] )", debug_str, re.DOTALL)
    return _parse_tensors(output_match.group(1)) if output_match else []


def _parse_tensor_info(dict_str):
    result = {}
    lines = dict_str.strip().split('\n')
    for line in lines:
        line = line.strip()
        if line and ':' in line:
            key, value = line.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key == "Place":
                place_match = re.search(r"Place\((\w+:\d+)\)", value)
                if place_match:
                    value = place_match.group(1)
            elif key == "Shape":
                value = [int(x.strip()) for x in value.split(',') if x.strip()]
            elif value == "None":
                value = None

            result[key] = value
    return result


def _split_list_elements(list_str):
    elements = []
    current = []
    brace_count = 0

    for char in list_str:
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1

        if char == ',' and brace_count == 0:
            elements.append(''.join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        elements.append(''.join(current).strip())

    return elements


def parse_debug_info(debug_str):
    result = {"API_Name": None, "Input": [], "Output": []}

    result["API_Name"] = _parse_api_name(debug_str)
    result["Input"] = _parse_input_tensors(debug_str)
    result["Output"] = _parse_output_tensors(debug_str)
    return result


class Edge:
    def __init__(self, tensor_info: dict, source=""):
        self.name = tensor_info["Name"]
        self.shape = tensor_info["Shape"]
        self.dtype = tensor_info["Dtype"]
        self.source = source

    def get_name(self) -> str:
        return self.name

    def get_source(self) -> str:
        return self.source

    def set_source(self, source):
        self.source = source

    def __str__(self) -> str:
        return f"Edge(name='{self.name}', source='{self.source}',shape='{self.shape}',dtype='{self.dtype}')"

    def get_edge_info(self):
        return f"{self.name}\nshape:{self.shape}\ndtype:{self.dtype}"


class Graph:
    def __init__(self):
        from graphviz import Digraph

        self.dot = Digraph()
        self.orange_box_attrs = {
            'style': 'rounded,filled,bold',
            'shape': 'box',
            'color': '#FFE4B5',
            'fillcolor': '#FFE4B5',
            'fontcolor': '#ffffff',
            'width': '1.3',
            'height': '0.84',
            'fontname': 'Arial',
        }
        self.grey_box_attrs = {
            'style': 'rounded,filled,bold',
            'shape': 'box',
            'color': '#999999',
            'fillcolor': '#999999',
            'fontcolor': '#ffffff',
            'width': '1.3',
            'height': '0.84',
            'fontname': 'Arial',
        }

        self.edges = {}
        self.nodes = []

    def add_node(self, name: str, node_attr=""):
        if not node_attr:
            node_attr = self.grey_box_attrs
        self.dot.node(name, name, **node_attr)
        self.nodes.append(name)

    # Link the src and dst node by edge
    def add_edge(self, dst: str, edge: Edge):
        edge_name = edge.get_name()
        if edge_name not in self.edges:
            # The Edge is not stored in the graph,
            # so the Edge's source node maybe not in the graph
            if not edge.get_source():
                src_name = edge.get_name() + "_SourceNode"
                edge.set_source(src_name)
                self.add_node(src_name, self.orange_box_attrs)
        else:
            edge = self.edges[edge_name]
        src = edge.get_source()
        self.dot.edge(src, dst, label=edge.get_edge_info())

    # Store an edge, not link the src and dst node
    def store_edge(self, edge: Edge):
        edge_name = edge.get_name()
        self.edges[edge_name] = edge

    def render(self, file_path):
        self.dot.render(file_path, format='svg')


class GraphBuilder:
    def __init__(self):
        self.graph = Graph()

    def build_graph(self, forward_debug_infos: list):
        for info in forward_debug_infos:
            debug_info = parse_debug_info(info)
            api_name = debug_info['API_Name']
            # Add a node for the API
            self.graph.add_node(api_name)
            # Store the Edge
            for out_param in debug_info["Output"]:
                var_name = out_param[0]
                tensors = out_param[1]
                for tensor_info in tensors:
                    # When we do not know the edge's dst, we should  store it to the Graph
                    edge = Edge(tensor_info, api_name)
                    self.graph.store_edge(edge)
            # Link the Edge
            for input_param in debug_info["Input"]:
                var_name = input_param[0]
                tensors = input_param[1]
                for tensor_info in tensors:
                    edge = Edge(tensor_info)
                    self.graph.add_edge(dst=api_name, edge=edge)

    def save_graph(self, file_path):
        self.graph.render(file_path)


@contextmanager
def capture_stderr():
    with tempfile.TemporaryFile(mode='w+b') as temp_file:
        original_stderr_fd = sys.stderr.fileno()
        saved_stderr_fd = os.dup(original_stderr_fd)
        stderr_output = io.StringIO()
        try:
            os.dup2(temp_file.fileno(), original_stderr_fd)
            sys.stderr.flush()
            yield stderr_output
        finally:
            sys.stderr.flush()
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.close(saved_stderr_fd)

            temp_file.seek(0)
            stderr_output.write(temp_file.read().decode())


@signature_safe_contextmanager
def capture_forward_subgraph_guard(file_path: str):
    log = ""
    stderr_buffer = io.StringIO()
    origin_enable_unique_name_status = paddle.framework.get_flags(
        "FLAGS_enable_unique_name"
    )["FLAGS_enable_unique_name"]
    try:
        paddle.set_flags({"FLAGS_enable_unique_name": True})
        # Redirect the stderr to the buffer,because the glog info will be printed to the stderr
        with (
            capture_stderr() as stderr_buffer,
            paddle.base.framework.vlog_guard(3),
        ):
            yield
    finally:
        paddle.set_flags(
            {"FLAGS_enable_unique_name": origin_enable_unique_name_status}
        )
        log = stderr_buffer.getvalue()
        builder = GraphBuilder()

        def get_first_indent(s):
            match = re.match(r'^\t*', s)
            return match.group(0)

        # Find the parts describing the input and output of the API in the massive logs
        indent = get_first_indent(log.lstrip('\n'))
        pattern_str = r'\n' + indent + r'Forward Debug Info \{.*? ] } '
        pattern = re.compile(pattern_str, re.DOTALL)
        matches = pattern.findall(log)
        # Build the forward graph
        builder.build_graph(matches)
        builder.save_graph(file_path)
