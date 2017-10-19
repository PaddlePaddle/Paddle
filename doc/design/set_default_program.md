# Design: Simply Layer program inputs

## Motivation

In the experience of writting simple python models, we found python API always need to specify a `program` and
a `init_program`. This is very redundant.

## Solution
```python
DEFAULT_PROGRAM = None
DEFAULT_INIT_PROGRAM = None

class Program(object):
    def __init__(self, name):
        self.name = name

class Model(object):
    def __init__(self, name):
        self.name = name
        self.p = Program("program")
        self.ip = Program("init_program")

    def __enter__(self):
        global DEFAULT_PROGRAM
        global DEFAULT_INIT_PROGRAM
        assert(DEFAULT_PROGRAM is None)
        assert(DEFAULT_INIT_PROGRAM is None)
        DEFAULT_PROGRAM = self.p
        DEFAULT_INIT_PROGRAM = self.ip

    def __exit__(self, type, value, traceback):
        global DEFAULT_PROGRAM
        global DEFAULT_INIT_PROGRAM
        DEFAULT_PROGRAM = None
        DEFAULT_INIT_PROGRAM = None

def fc_layer():
    DEFAULT_PROGRAM.name += " fc_layer added"
    DEFAULT_INIT_PROGRAM.name += " fc_layer added"

def main():
    m = Model("M")

    print(m.p.name)
    print(m.ip.name)

    with m:
        fc_layer()

    print(m.p.name)
    print(m.ip.name)

if __name__ == "__main__":
    main()
```
It prints
```
program
init_program
program fc_layer added
init_program fc_layer added
```
