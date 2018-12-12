from ....framework import Program

__all__ = ['Graph', 'ImitationGraph', 'IRGraph']

class Graph(object):
    """
    Base class for all graph.
    """
    def __init__(self):
        pass

    def all_parameters(self):
        """
        Return all the parameters in current graph.
        """
        pass


class ImitationGraph(Graph):
    def __init__(self, program=None):
        super(ImitationGraph, self).__init__()
        self.program = Program() if program is None else program

    def all_parameters(self):
        return self.program.global_block().all_parameters()

    
class IRGraph(Graph):
    pass
    
