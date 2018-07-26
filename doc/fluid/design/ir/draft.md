## Motivation

There is a `gap` between the `Program` defined by
user and the `Executable` that can be scheduled
efficiently on heterogeneous hardware, either locally
or distributedly.

Usually, the `gap` is bridged by

* A serious transformations with defined order.

* These transformations usually involve
`insert, delete, clustering, split, dependency analysis`.

* Has a simple way to verify and debug each transformation.

* Flexible to add, remove or customize transformations to fit
the requirements of various algorithms (models) and hardware secenarios.

Some other events also push us to a better unified pattern.

* The deep learning framework is built around the concepts of graphs.
To leverage tools such as compilation (e.g. TVM and nGraph) or
cross-framework conversion (e.g. ONNX), we also need a intermediate
representation that can be connected to the rest of the ecosystem.


We need a unified pattern to naturally support the requirements
described above. The pattern should fit both training, inference
and other offline serielized model transformations.
Learned from LLVM and other deep learning framework, we draft the
design below.


## Design

### Major Concepts

#### Node

`Node` represents an operation that performs some computation or
a variable that is input or output of operation.

`Node`s are connected to other `Node`s via inputs and outputs.

Other properties (maybe device placement information) can be added
to `Node` in the future if it's a
common requirement of many other `Pass`es. Otherwise, it should live
in a `Node` wrapper class that is private to some `Pass` or be
a local member of a `Pass`.

#### Graph

`Graph` contains a list of `Node`s, which are connected to
each other via inputs and outputs.

TODO: Better definitions for the graph.

`Graph` can also contain `Attribute`s. `Attribute`s
can be `any` thing. For example, it can be a list of "wraper"
nodes. The `wrapper` nodes compose `Node`s and provide
helper method for execution or transformation. `Attribute`
can also contain other things that describe some properties of
the `Graph` or `Graph` nodes. `Attribute` can be passed
across `Pass`. However, it should be used with care.

#### Pass

`Pass` represents a transformation of `Graph`. Its input
is a `Graph` and its output is also a `Graph`. For example,
a `Pass` can simply print out the `Graph`. A `Pass`
can also fuse some `Graph`'s `Node`s.

```cpp
class Pass {
 public:

  virtual std::unique_ptr<Graph> Apply(std::unique_ptr<Graph> graph) const = 0;

  // Get a reference to the attributed previously set.
  template <typename AttrType>
  AttrType &Get(const std::string &attr_name) const;

  // Set a pointer to the attribute. Pass takes ownership of the attribute.
  template <typename AttrType>
  void Set(const std::string &attr_name, AttrType *attr) ;

  // Set a pointer to the attribute. Pass doesn't take ownership. Caller
  // should delete the attribute.
  template <typename AttrType>
  void SetNotOwned(const std::string &attr_name, AttrType *attr);
};

// In my_pass.cc
class MyPass : public Pass {
 public:
  std::unique_ptr<Graph> Apply(std::unique_ptr<Graph> graph) const override {
    // do something.
    return graph;
  }
}
REGISTER_PASS(my_pass, MyPass);


// To use the pass.
auto my_pass = ir::PassRegistry::Instance().Get("my_pass");
graph = my_pass->Apply(std::move(graph));
// Note: to force link my_pass.cc, in the code:
USE_PASS(my_pass);
```

#### Optimize

`Optimize` contains a series of `Pass` with defined order.
`Optimize` transforms a `Graph` that only contains raw
modeling logic to a `Graph` that can be run efficiently while
maintaining the original modeling logic.


### Optimize Process

* Program is first converted to Graph.
* Graph goes through a series of Pass
* Graph is transformed from raw model logic to a
form that is efficient to execute.

Program->ProgramToGraph->Graph->Pass1->Graph->Pass2->Graph->Pass3->Graph->Executor
