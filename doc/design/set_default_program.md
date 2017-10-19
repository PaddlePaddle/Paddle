```python
DEFAULT_PROGRAM = None
DEFAULT_INIT_PROGRAM = None

class Program:
    def __init__(self, name):
        self.name = name

class set_program_desc:
    def __init__(self, p, ip):
        self.p = p
        self.ip = ip

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
    p = Program("program")
    ip = Program("init_program")
    print(p.name)
    print(ip.name)

    with set_program_desc(p, ip):
        fc_layer()

    print(p.name)
    print(ip.name)

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
