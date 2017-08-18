class Graph(object):
    def __init__(self):
        # node : str
        # targets: list
        self.edges = {}

    def add(self, op, reverse=False):
        '''
        add edges according to op's inputs and outputs
        '''
        sources = op.inputs if not reverse else op.outputs
        targets = op.outputs if not reverse else op.inputs
        for var in sources:
            key = repr(var)
            if key not in self.edges:
                self.edges[key] = []
            self.edges[key] += [repr(op)]
        for var in targets:
            key = repr(op)
            if key not in self.edges:
                self.edges[key] = []
            self.edges[key] += [repr(var) for var in op.outputs]

    def __str__(self):
        pass


class Topology(object):
    '''
    graph generator
    '''

    def __init__(self, session):
        self.session = session
        self.graph = Graph()
        self.reverse_graph = Graph()

    def build(self):
        self._create_graph()
        self._create_reverse_graph()

    def _create_graph(self):
        for op in self.session.ops:
            self.graph.add(op)

    def _create_reverse_graph(self, graph):
        for op in self.session.ops:
            self.graph.add(op, reverse=True)

    def DFS_to_targets(self, targets=[]):
        '''
        @targets: list of Val
        return: reprs of vars and ops
        '''
        visited_set = set()
        visited_nodes = []

        # use reverse graph, the targets as the start points
        # get the sequence visited, reverse it and return

        stack = targets
        cur = None
        for target in targets:
            while stack:
                cur = stack.pop()
                if cur not in visited_set:
                    visited_set.add(cur)
                    visited_nodes.append(cur)
                    stack.extend(self.graph.edges[cur])
        return list(reversed(visited_nodes))
