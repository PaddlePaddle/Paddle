from layer import Layer
from op import Op
from variable import Variable

class DependenceGraph(object):
    def __init__(self, nodes):
        '''
        nodes: list of Variable, Operator or Layer

        reversed: bool
        whether to generate a reversed graph, target->source
        '''
        # source -> list of targets
        self.edges = {}
        self.nodes = nodes
        self._build()

    def DFS_with_targets(self, targets, filter_type=[Op, Layer]):
        '''
        targets: list
        '''
        visited = set()
        visited_nodes = []

        # TODO(superjom) need a copy here?
        stack = targets
        cur = None

        for target in targets:
            while stack:
                cur = stack.pop()
                if cur not in visited:
                    visited.add(cur)
                    visited_nodes.append(cur)
                    stack.append(self.graph.edges[cur])

        visited_nodes = filter(
            lambda _: any(isinstance(_, t) for t in filter_type),
            visited_nodes)

        return visited_nodes

    def _build(self):
        assert not self.edges, "graph can't be built more than once"
        for node in self.nodes:
            if not isinstance(node, Op):
                continue
            # op -> input
            for input in node.inputs:
                if node not in self.edges:
                    self.edges[node] = set()
                self.edges[node].add(input)
            # output -> op
            for output in node.outputs:
                if output not in self.edges:
                    self.edges[output] = set()
                self.edges[output].add(node)


