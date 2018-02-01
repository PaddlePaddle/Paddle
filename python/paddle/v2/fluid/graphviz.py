#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import random
import subprocess


def crepr(v):
    if type(v) is str or type(v) is unicode:
        return '"%s"' % v
    return str(v)


class Rank(object):
    def __init__(self, kind, name, priority):
        '''
        kind: str
        name: str
        priority: int
        '''
        self.kind = kind
        self.name = name
        self.priority = priority
        self.nodes = []

    def __str__(self):
        if not self.nodes:
            return ''

        return '{' + 'rank={};'.format(self.kind) + \
               ','.join([node.name for node in self.nodes]) + '}'


class Graph(object):
    rank_counter = 0

    def __init__(self, title, **attrs):
        self.title = title
        self.attrs = attrs
        self.nodes = []
        self.edges = []
        self.rank_groups = {}

    def code(self):
        return self.__str__()

    def rank_group(self, kind, priority):
        name = "rankgroup-%d" % Graph.rank_counter
        Graph.rank_counter += 1
        rank = Rank(kind, name, priority)
        self.rank_groups[name] = rank
        return name

    def node(self, label, prefix, **attrs):
        node = Node(label, prefix, **attrs)

        if 'rank' in attrs:
            rank = self.rank_groups[attrs['rank']]
            del attrs['rank']
            rank.nodes.append(node)
        self.nodes.append(node)
        return node

    def edge(self, source, target, **attrs):
        edge = Edge(source, target, **attrs)
        self.edges.append(edge)
        return edge

    def compile(self, dot_path):
        file = open(dot_path, 'w')
        file.write(self.__str__())
        image_path = dot_path[:-3] + "pdf"
        cmd = ["dot", "-Tpdf", dot_path, "-o", image_path]
        subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        return image_path

    def show(self, dot_path):
        image = self.compile(dot_path)
        cmd = ["open", image]
        subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)

    def _rank_repr(self):
        ranks = sorted(
            self.rank_groups.items(),
            cmp=lambda a, b: a[1].priority > b[1].priority)
        repr = []
        for x in ranks:
            repr.append(str(x[1]))
        return '\n'.join(repr) + '\n'

    def __str__(self):
        reprs = [
            'digraph G {',
            'title = {}'.format(crepr(self.title)),
        ]

        for attr in self.attrs:
            reprs.append("{key}={value};".format(
                key=attr, value=crepr(self.attrs[attr])))

        reprs.append(self._rank_repr())

        random.shuffle(self.nodes)
        reprs += [str(node) for node in self.nodes]

        for x in self.edges:
            reprs.append(str(x))

        reprs.append('}')
        return '\n'.join(reprs)


class Node(object):
    counter = 1

    def __init__(self, label, prefix, **attrs):
        self.label = label
        self.name = "%s_%d" % (prefix, Node.counter)
        self.attrs = attrs
        Node.counter += 1

    def __str__(self):
        reprs = '{name} [label={label} {extra} ];'.format(
            name=self.name,
            label=self.label,
            extra=',' + ','.join("%s=%s" % (key, crepr(value))
                                 for key, value in self.attrs.items())
            if self.attrs else "")
        return reprs


class Edge(object):
    def __init__(self, source, target, **attrs):
        '''
        Link source to target.
        :param source: Node
        :param target: Node
        :param graph: Graph
        :param attrs: dic
        '''
        self.source = source
        self.target = target
        self.attrs = attrs

    def __str__(self):
        repr = "{source} -> {target} {extra}".format(
            source=self.source.name,
            target=self.target.name,
            extra="" if not self.attrs else
            "[" + ','.join("{}={}".format(attr[0], crepr(attr[1]))
                           for attr in self.attrs.items()) + "]")
        return repr


g_graph = Graph(title="some model")


def add_param(label, graph=None):
    if not graph:
        graph = g_graph
    return graph.node(label=label, prefix='param', color='blue')


def add_op(label, graph=None):
    if not graph:
        graph = g_graph
    label = '\n'.join([
        '<table border="0">',
        '  <tr>',
        '    <td>',
        label,
        '    </td>'
        '  </tr>',
        '</table>',
    ])
    return graph.node(label=label, prefix='op', shape="none")


def add_edge(source, target):
    return g_graph.edge(source, target)


class GraphPreviewGenerator(object):
    '''
    Generate a graph image for ONNX proto.
    '''

    def __init__(self, title):
        # init graphviz graph
        self.graph = Graph(
            title,
            layout="dot",
            concentrate="true",
            rankdir="TB", )

        self.op_rank = self.graph.rank_group('same', 2)
        self.param_rank = self.graph.rank_group('same', 1)
        self.arg_rank = self.graph.rank_group('same', 0)
        # self.nodes = {}

    def __call__(self, path='temp.dot', show=False):
        if not show:
            self.graph.compile(path)
        else:
            self.graph.show(path)

    def add_param(self, name, data_type, shape):
        label = '\n'.join([
            '<<table cellpadding="5">',
            '  <tr>',
            '    <td bgcolor="#2b787e">',
            '    <b>',
            name,
            '    </b>',
            '    </td>',
            '  </tr>',
            '  <tr>',
            '    <td>',
            str(data_type),
            '    </td>'
            '  </tr>',
            '  <tr>',
            '    <td>',
            '[%s]' % 'x'.join(shape),
            '    </td>'
            '  </tr>',
            '</table>>',
        ])
        return self.graph.node(
            label,
            prefix="param",
            shape="none",
            style="rounded,filled,bold",
            width="1.3",
            color="#148b97",
            fontcolor="#ffffff",
            fontname="Arial")

    def add_op(self, opType, **kwargs):
        return self.graph.node(
            "<<B>%s</B>>" % opType,
            prefix="op",
            shape="box",
            style="rounded, filled, bold",
            color="#303A3A",
            fontname="Arial",
            fontcolor="#ffffff",
            width="1.3",
            height="0.84", )

    def add_arg(self, name):
        return self.graph.node(
            crepr(name),
            prefix="arg",
            shape="box",
            style="rounded,filled,bold",
            fontname="Arial",
            fontcolor="#999999",
            color="#dddddd")

    def add_edge(self, source, target, **kwargs):
        # source = self.nodes[source]
        # target = self.nodes[target]
        return self.graph.edge(source, target, **kwargs)


if __name__ == '__main__':
    n0 = add_param(crepr("layer/W0.w"))
    n1 = add_param(crepr("layer/W0.b"))

    n2 = add_op("sum")

    add_edge(n0, n2)
    add_edge(n1, n2)

    print g_graph.code()
    g_graph.compile('./1.dot')
