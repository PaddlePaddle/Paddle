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
class AddUnaryOp:
    def __init__(self, data):
      self.data = data

@dataclass
class AddBinaryOp:
    def __init__(self, data):
      self.data = data

@dataclass
class InsertBinaryOp:
    def __init__(self, data):
      self.data = data

@dataclass
class AddBinaryClone:
    def __init__(self, data):
      self.data = data

@dataclass
class AddSourceOp:
    def __init__(self, data):
      self.data = data

IrGenType = ( Nope
            | AddSinkTensor
            | AddUnaryOp
            | AddBinaryOp
            | InsertBinaryOp
            | AddBinaryClone
            | AddSourceOp
            )