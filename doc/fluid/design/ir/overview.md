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

```cpp
class Graph {
 public:
  explicit Graph(const ProgramDesc &program);

  bool Has(const std::string &attr_name) const;

  template <typename AttrType>
  AttrType &Get(const std::string &attr_name) const;

  template <typename AttrType>
  void Set(const std::string &attr_name, AttrType *attr);
  const std::unordered_set<ir::Node *> &Nodes() const;

  // Create a normal variable with non-null VarDesc.
  ir::Node *CreateVarNode(VarDesc *var_desc);

  // Create a normal runnable operator with OpDesc.
  ir::Node *CreateOpNode(OpDesc *op_desc);

  // Create a control dependency var that connects 2 operations. The
  // var doesn't hold any data. Other than that, it's no different from
  // other var, considering dependency analysis.
  ir::Node *CreateControlDepVar();

  // A more free style way of creating a graph node. Mostly use for test
  // or "copy" from another node. Avoid using it if possible.
  ir::Node *CreateEmptyNode(const std::string &name, ir::Node::Type type);

  // Clear all node information of the graph and return the ownership of the
  // nodes.
  std::vector<std::unique_ptr<ir::Node>> ReleaseNodes();
};
```

#### Pass

`Pass` represents a transformation of `Graph`. Its input
is a `Graph` and its output is also a `Graph`. For example,
a `Pass` can simply print out the `Graph`. A `Pass`
can also fuse some `Graph`'s `Node`s.

```cpp
class Pass {
 public:

  std::unique_ptr<Graph> Apply(std::unique_ptr<Graph> graph) const {
    // Some correctness check.
    auto new_graph = ApplyImpl(std::move(graph));
    // Some correctness check.
    return new_graph;
  }

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

 protected:
  virtual std::unique_ptr<Graph> ApplyImpl(std::unique_ptr<Graph> graph) const = 0;
};

// In my_pass.cc
class MyPass : public Pass {
 protected:
  std::unique_ptr<Graph> ApplyImpl(std::unique_ptr<Graph> graph) const override {
    // do something.
    return graph;
  }
}
REGISTER_PASS(my_pass, MyPass)
.RequirePassAttr("places")
.RequireGraphAttr("dep_vars");


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

```
// Program->ProgramToGraph->Graph->Pass1->Graph->Pass2->Graph->Pass3->Graph->Executor
auto graph = Graph(program);
graph = PassRegistry::Instance().Get("op_fuse_pass").Apply(std::move(grah));
// For more complex Pass, Optimize Process can provide Pass attributes.
auto mem_opt_pass = PassRegistry::Instance().Get("memory_optimization_pass");
mem_opt_pass.SetNotOwned<int>("optimize_level", 1);
mem_opt_pass->Apply(std::move(graph));
graph = PassRegistry::Instance().Get("multi_devices_pass").Apply(std::move(grah));
graph = PassRegistry::Instance().Get("multi_devices_check_pass").Apply(std::move(grah));
Executor exe;
exe.Run(graph);

```
