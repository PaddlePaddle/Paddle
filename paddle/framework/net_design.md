# Network Design

`Network` is the container and controller of a set of operators in a network, 
users can build a real network from a `NetDef` in protobuf message 
and use `Network.Run()` to run all the operators in the network.

The `Network` will

- manage all the operators contained in the network.
- not own any `Variable`.

# API

## NetworkBase
To make the `Network` extendable, a base class is defined like this

```c++
// The minimum a network should be implemented.
class NetworkBase {
 public:
  // `def` is a proto message that describe the structure of a network.
  NetworkBase(const NetDef& def);

  // run all the operators and return success(true) or not, all the
  // variables are located in `scope`.
  virtual bool Run(Scope* scope) = 0;

 protected:
  // keys of the input variables feed into the network.
  std::vector<string> inputs_;
  // keys of the corresponding output variables the network will mutate.
  std::vector<string> outputs_;
};
```

All network implementations should build networks from  a protobuf message which 
describes the structure of a real network; `Run` method should be implemented by 
all implementations to offer a universal method to forward or backward compute a network.

A method of factory pattern can be defined like

```c++
NetworkBase* CreateNet(const NetDef& def) {
  switch (def.model_type()) {
    case NN:
      return new Network(def);
    case Recursive:
      return new RecursiveNet(def);
    case Recurrent:
      return new RecurrentNet(def);
  }
  return new Network(def);
}
```

Network is designed as the container of operators, to make it more extendable,
we decompling it from the related variable resources. 
A `scope` is provided to `Run` so that the network structure can be reused 
in different scopes.

Finally, `NetworkBase` can be used as followed

```c++
Scope default_scope;
auto net = CreateNet(def);
net.Run(&default_scope);
```


## A Simple Network Implemention

A simple implemention is as followed:

```c++
class ScratchNet final : public NetworkBase {
 public:
  // Create a network describe by `def`.  NetDef is the definition of a network.
  ScratchNet(const NetDef &def);

  // Run all the operators with the `scope`, if no scope is provided, default
  // scope will be used instead.
  virtual bool Run(Scope *scope = nullptr) override;

 protected:
  // Create operators accordding to `def`.
  bool CreateNet(const NetDef &def);

  // Add a operator which is identified as `type` and has attributes described
  // in `attrs`, the `inputs` are the keys of readonly input variables,
  // `outputs` are keys of mutable output variables.
  bool AddOp(const std::string &type, const std::vector<string> &inputs,
             const std::vector<string> &outputs,
             const OprAttr &attrs = OprAttr());

 private:
  // the operators owned by `Network`.
  std::vector<std::unique_ptr<Operator>> ops_;
};
```

`ScratchNet` will create operators so that a private member `ops_` is defined,
the operators are created by `CreateNet`, and each operator is created by `AddOp`.



## Usage
`ScratchNet` can be used to define and run a network as followed

```c++
// create an empty scope located on CPU device.
Scope scope(CPUPlace());

// create and init variables described in `net_desc`.
scope.CreateVariables(net_desc);
scope.InitVariables(net_desc);

// create a network according to `net_desc`
auto net = CreateNet(net_desc);

// run the network providing the `scope`.
net.Run(&scope);
```

## Compatibility with RNN

Benefit from the decoupling of `Run` and `Scope`, `ScratchNet` is compatible with RNN design, 
for example we can implement a simple recurrent neural network as followed

```c++
// copy some `vars` form `source` to `target`
void Copy(const Scope &source, Scope &target,
          const std::vector<std::string> &vars);

Scope default_scope;
// some initial mutations on `default_scope` here.

auto rnn_net = ScratchNet(rnn_net_def);

// Create rnn's states, the last scope is used to store rrn's outputs.
Scope *rnn_states = new Scope[num_states + 1];

for (int i = 0; i < num_states + 1; i++) {
  // Initialize all rnn state scopes, copy parameters and so on.
  rnn_states[i].CreateVars(rnn_net_def);
  Copy(default_scope, rnn_states[i], rnn_related_vars);
  // Prepare rnn's inlinks, just copy inlink variables to each state.
  Copy(default_scope, rnn_states[i], inlink_vars);
}

// Run the rnn.
for (int i = 0; i < num_states; i++) {
  rnn_net.Run(rnn_states[i]);
  // Copy current state's state variables to next state, the related variables
  // are named like "previous_state_xxx".
  Copy(rnn_states[i], rnn_states[i + 1], pre_state_vars)
}

// Copy rnn's final outputs to `default_scope`.
Copy(rnn_states[num_states], default_scope, outlink_vars);
```
