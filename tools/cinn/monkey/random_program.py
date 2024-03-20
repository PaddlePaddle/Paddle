from dataclasses import dataclass

@dataclass
class Nope:
    def __init__(self):
      pass

@dataclass
class AddSinkTensor:
    def __init__(self, data):
      self.data = data

@dataclass
class AddUnaryUpstreamOp:
    def __init__(self, data):
      self.data = data

@dataclass
class AddBinaryUpstreamOp:
    def __init__(self, data):
      self.data = data

@dataclass
class InsertBinaryUpstreamOp:
    def __init__(self, data):
      self.data = data

@dataclass
class AddBinaryCloneUpstream:
    def __init__(self, data):
      self.data = data

@dataclass
class MarkFinalSourceTensor:
    def __init__(self, data):
      self.data = data

IrGenType = ( Nope
            | AddSinkTensor
            | AddUnaryUpstreamOp
            | AddBinaryUpstreamOp
            | InsertBinaryUpstreamOp
            | AddBinaryCloneUpstream
            | MarkFinalSourceTensor
            )