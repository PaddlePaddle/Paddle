Overview Design


In the refactoring of Paddle, we want to change the computation from layer based to operator based. The main reasons are:

1. operator can represent more fine grit function.
1. separate computation with data. Operator can focus on the computation only.
1. can represent more complex graph such as backward in backward.

Firstly we need a structure to describe the Graph, that's `Network`.


## CPP end

### Network
Network is equal to computation Graph. We use a sequence of Operator to describe the computation graph and drive the computation. There are two most important component in a Network.

1. Operator
1. Variable

Network call the Run interface of Operator to do some computing, Operator will fetch the related Variables from somewhere, to store this variables, we will have another component:

1. Scope


The Computation not only need data in memory, but also need some computing resource such as CPU/GPU/BLASLIB, to manage this computing resource, we will have one more componet:

1. Context

The basic interface is:

```cpp
class Network {
public:
 Error Run() {
   for(op in ops_) {
    op.Run(scope, context)
   }
 }

proviate:
 vector<Operators> ops_;
}
```

### Operator

Operator is something like a function. It takes some Inputs and produced some Outputs.

To process and represent all kinds of Inputs/Outputs, we provide a data container:

1. Variable

Different Operators may have different parameters such as `scale` for `cosine` Operator, to describe these we have:

1. OpAttr

so the base define of Operator look like:
```cpp
class Operator {
public:
    ERROR Run(vector<Variable> inputs, vector<Variable> outputs) {
    // computation logic
    }

private:
    vector<OpAttr> opAttrs;
}
```

### Variable
Variable is actually a typed pointer that can store any type of C++ objects, but Tensor is the most common type stored in a Variable.


#### OpAttr
OpAttr is used to store some common data types for operator to use when computing, such as hyper-parameters. We use a proto message to represent them.

```proto
message AttrDesc {
  AttrType type = 1;
  optional int i = 2;
  optional float f = 3;
  optional string s = 4;
  repeated int ints = 5;
  repeated float floats = 6;
  repeated string strings = 7;
};
```

### Scope
Scope manage Variables you create and store in memory. You can create or get a Variable from a Scope. The basic interface for a scope is

```cpp
class Scope {
public:
    Variable* CreateVar(string name);
    Variable* GetVar(string name);
}
```

## Python end

We want to use python as to construct the computation Graph and drive the compute, but all `Operators` are in cpp, we don't want to write a wrapper for every Operator.
So we want a method to `generate` python wrappers for `Operators`. To do this, we need a structure to describe all Operators and python can generate op wrapper by reading this message.

So we use a proto message `OperatorProto` to do this.

### OperatorProto
The most important things for Operator is `input/output` and `attr`:

```proto
message VarProto {
  required string name = 1;
  required string comment = 2;
};
enum AttrType {
  INT = 1,
  FLOAT = 2,
  STRING = 3,
  INTS = 4,
  FLOATS = 5,
  STRINGS = 6
}
message AttrProto {
  required string name = 1;
  required string comment = 2;
  required AttrType type = 3;
};

message OpProto {
  repeated VarProto inputs = 1;
  repeated VarProto outputs = 2;
  required string comment = 3;
  repeated AttrProto attrs = 4;
  required string type = 5;
};
```

All Ops should fill this structure and register it to a registry in CPP, so we need a OperatorRegistry to register and store all this information and
provide python interface to get them all.

### OperatorRegistry


### Model

operators and variables are two low level for user to construct the network. To add some

