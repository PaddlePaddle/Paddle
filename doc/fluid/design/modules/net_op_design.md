# Network Design

`Network` is the container and controller of a set of operators,
user can build a real network from a `NetDesc` which is a protobuf message
and use `Network.Run()` to run all the operators in the network.

A network object knows all Operators belonging to this network. Variables,
which are inputs and outputs of these operators,
are created and managed by a hierarchy of Scope objects.

## API

### Net
To make the `Network` extendable, a base class is defined like this

```c++
// operator's index stored in a network.
typedef int OpIndex;

// The minimum a network should be implemented.
class Net {
 public:
  // run all the operators and return success(true) or not, with all the
  // variables are located in `scope`. `context` describes the detail execution
  // environment for ops. `begin` and `end` specify the scope of `ops_` to run,
  // If no positive indexes are provided, all operators in `ops_` will run.
  virtual Error Run(Scope *scope, OpContext *context, OpIndex begin = -1,
                   OpIndex end = -1) const = 0;

  // Add an Operator according to `def`.
  virtual OpIndex AddOp(const proto::OpDef &def) = 0;

  // Add optimizer operators acctording to `attrs`.
  virtual Error AddOptimizerOps(const OptAttrs &attrs) = 0;

  // Add backward operators.
  virtual Error AddBackwardOps() = 0;

  // Infer the shapes of variables required by operators in the network. The
  // `scope` will be mutated according to the inferred shapes.

  static std::unique_ptr<Net> Create(const NetDesc &def = NetDesc());
};
```

All network implementations should build networks from a protobuf message which
describes the structure of a real network; `Run` method should be implemented by
all implementations to offer a universal method to forward or backward compute a network.

`Net::Create` is a method of factory pattern and can be implemented like

```c++
std::unique<Net> Net::Create(const NetDesc& def) {
  switch (def.model_type()) {
    case NN:
      return new Network(def);
    case Recursive:
      return new RecursiveNet(def);
    case Recurrent:
      return new RecurrentNet(def);
  }
  return nullptr;
}
```

Network is designed as the container of operators. to make it more extendable,
we decouple it from the related variable resources.

`Run(Scope* scope)` takes the scope as a argument so that it can run in different scopes.

Finally, `Net` can be used as followed

```c++
Scope default_scope;
OpContext default_context;
auto net = Net::CreateNet(def);

if (net) {
  net.Run(&default_scope, &default_context);
}
```

### `PlainNet` as a simple implementation of `BaseNet`

A very basic implementation is as follows. All it does is simply to run every operators in sequence.

```c++
class PlainNet : public Net {
 public:
  // Create a network describe by `def`.  NetDesc is the definition of a network.
  PlainNet(const NetDesc &def);

  // Infer all the operators' input and output varialbes' shapes, will be called before every mini-batch
  training.
  virtual Error InferShape(Scope *scope) override;

  // Run all the operators with the `scope`, if no scope is provided, default
  // scope will be used instead. If no OpContext is provicded, default context will be used.
  virtual Error Run(Scope *scope = nullptr, OpContext *context=nullptr, OpIndex begin = -1,
                   OpIndex end = -1) const override;

  virtual OpIndex AddOp(const proto::OpDef &def) override;

  virtual Error AddOptimizerOps(const OptAttrs &attrs) override;

  virtual Error AddBackwardOps() override;

 protected:
  // Create operators accordding to `def`, will be called by the constructor.
  Error BuildNet(const NetDesc &def);

  // Add a operator which is identified as `type` and has attributes described
  // in `attrs`, the `inputs` are the keys of readonly input variables,
  // `outputs` are keys of mutable output variables. An `OpIndex` will be
  // returned to indicate the offset of the new operator in `ops_`.
  OpIndex AddOp(const std::string &type, const std::vector<string> &inputs,
                const std::vector<string> &outputs,
                const OprAttr &attrs = OprAttr());

 private:
  // the operators owned by `Network`.
  std::vector<Operator> ops_;
};
```

`PlainNet` will create operators so that a private member `ops_` is defined,
the operators are created by `CreateNet`, and each operator is created by `AddOp`.


## PlainNet Usage
`PlainNet` can be used to define and run a network as follows

```c++
// create an empty scope located on CPU device.
Scope scope(CPUPlace());

// create and init variables described in `net_desc`.
scope.CreateVariables(net_desc);
scope.InitVariables(net_desc);

// create a network according to `net_desc`
auto net = Net::CreateNet(net_desc);
// Add more operators if needed.
net->AddOp(add...);
net->AddOp(fc...);

net->AddBackwardOps();
net->AddOptimizerOps();

// run the network providing the `scope`.
net.Run(&scope);
```

## `NetBuilder` as a C++ syntax wrapper
This is a detailed description of the user-related C++ network API, and may not needed in the prototype development stage.

The `NetBuilder` will give users a much simpler syntax as follows to create a network, and demonstrates how to use the `BaseNet`'s raw interfaces.

```c++
Variable* fc_out = builder.AddOp("fc", input=image, size=100, activation="Sigmoid");
Variable* prediction = builder.AddOp("fc", input=fc_out, size=10, activation="Sigmoid");
Variable* loss = builder.AddOp("cross_entropy", input=prediction, label=label);
Variable* avg_loss = builder.AddOp("mean", loss);

builder.BackwardFrom(avg_loss)
builder.AddOptimization(1e-4, "adam");
builder.Run();
```

`NetBuilder` will call `Net` 's virtual functions to change the real network structure, here is a sample definition

```c++
class NetBuilder final {
 public:
  NetBuilder(Net* net) : net_(net) {}

  Variable* AddOp(const string& type, const vector<Variable>& inputs,
                  size_t size, Activation act) {
    // much code here.
    // ...
    net_->AddOp(def);
    need_rebuild_net_ = true;
    net_->InferShape();
    // ...
  }

  Error BackwardFrom(const Variable& cost);

  Error Run(Scope* scope, OpContext* context, bool need_backward = true) {
    // backward.
    if (need_backward) {
      if (need_rebuild_net_) {
        AddBackwardOps();
        AddOptimizerOps();
      }
      net_->Run(scope, context);
      return;
    }
    // just forward.
    net_->Run(scope, context, 0, last_forward_op_);
  }

 protected:
  Error AddBackwardOps();
  Error AddOptimizerOps();

 private:
  Net* net_;
  OpIndex last_forward_op_{-1};
  bool need_rebuild_net_{true};
}
```

### Compatibility with RNN

Benefitting from the decoupling of `PlainNet.Run` and `Scope`, `PlainNet` is compatible with future RNN design,
for example we can implement a simple recurrent neural network as follows

```c++
// copy some `vars` form `source` to `target`
void Copy(const Scope &source, Scope &target,
          const std::vector<std::string> &vars);

Scope default_scope;
// some initial mutations on `default_scope` here.

auto rnn_step_net = PlainNet(rnn_step_net_def);

// Create rnn's states, the last scope is used to store rnn outputs.
Scope *rnn_states = new Scope[num_states + 1];

for (int i = 0; i < num_states + 1; i++) {
  // Initialize all rnn state scopes, copy parameters and so on.
  rnn_states[i].CreateVars(rnn_step_net_def);
  Copy(default_scope, rnn_states[i], rnn_related_vars);
  // Prepare rnn's inlinks, just copy inlink variables to each state.
  Copy(default_scope, rnn_states[i], inlink_vars);
}

// Run the rnn.
for (int i = 0; i < num_states; i++) {
  rnn_step_net.Run(rnn_states[i]);
  // Copy current state's state variables to next state, the related variables
  // are named like "previous_state_xxx".
  Copy(rnn_states[i], rnn_states[i + 1], pre_state_vars)
}

// Copy rnn's final outputs to `default_scope`.
Copy(rnn_states[num_states], default_scope, outlink_vars);
```
