# Network Design

`Network` is the container and controller of a set of operators in a network, users can use `Network.AddOp` to add operators into a network, 
and use `Network.Run()` to run all the operators in the network.

The `Network` will

- manage all the operators contained in the network.
- not own any `Variable`.

# API

To make the `Network` extendibe, a base class is defined like this

```c++
// The minimum a network should be implemented.
class NetworkBase {
public:
  NetworkBase(const NetDef &def);

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

A simple implemention is as followed:

```c++
class Network final : public NetworkBase {
 public:
  // Create a network describe by `def`.  NetDef is the definition of a network.
  Network(const NetDef &def);

 protected:
  // Add a operator which is identified as `type` and has attributes described
  // in `attr`, the `inputs` are the keys of readonly input variables, `outputs`
  // are keys of mutable output variables.
  bool AddOp(const std::string &type, const std::vector<string> &inputs,
             const std::vector<string> &outputs,
             const OprAttr &attr = OprAttr());

  // Run all the operators with the `scope`, if no scope is provided, default
  // scope will be used instead.
  virtual bool Run(Scope *scope = nullptr) override;

private:
  // the operators are owned by `Network`.
  std::vector<std::unique_ptr<Operator>> ops_;
};
```

We can define and run a network like this

```c++
// create an empty scope located on CPU device.
Scope scope(CPUPlace());

// create and init variables described in `net_desc`.
w1.CreateVariables(net_desc);
w1.InitVariables(net_desc);

// create a network according to `net_desc`
auto net = CreateNet(net_desc);

// run the network providing the `scope`.
net.Run(&scope);
```
