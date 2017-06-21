# Network Design

`Network` is the container and controller of a set of operators in a network, users can use `Network.addOp` to add operators into a network, 
and use `Network.runOps` to run all the operators in the network.

The `Network` will

- manage all the operators contained in the network.
- not own any `Variable`.

# API

To make the `Network` extendibe, a base class is defined like this

```c++
// The minimum a network should be implemented.
class BaseNetwork {
public:
  BaseNetwork(const NetDef& def, Scope *scope);

  // run all the operators and return success(true) or not.
  virtual bool Run() = 0;

protected:
  // the input variables feed into the network.
  std::vector<Variable*> external_inputs_;
  // the corresponding output variables the network will write.
  std::vector<Variable*> external_outputs_;
  // scope which contains all the global variables visiable to this network.
  Scope *scope_;
};
```

A simple implemention is as followed:

```c++
class Network : public BaseNetwork {
public:

  // Create an empty network. 
  Network(const std::string& name, Scope *scope);
  
  // NetDef is the definition of a network, in some occasion, operators are created 
  // dynamically by user one by one; but in some other occasion such as LSTM, all 
  // the operators in the networks should be  created during the construction 
  // of the network. So a `NetDef` is provided to make the `Network` create a 
  // network with all the operators described in `def`.
  Network(const std::string& name, const NetDef& def, Scope *scope);
  
  // add a operator which is identified  as `type` and has attributes described
  // in `attr`.
  bool AddOp(const std::string &type, const OprAttr& attr);
  
  // run all operators in oprs_ sequentially.
  virtual bool Run() override;
  
protected:
  // to make the network's structure more human-readable, each network will 
  // has a `name`
  std::string name_;
  // the operations are owned by `Network`.
  std::vector<std::shared_ptr<Operator>> oprs_;
};
```

