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
  BaseNetwork(const NetDef &def, Scope *scope);

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
class Network final : public BaseNetwork {
public:
  // Create an empty network, user can add operators by calling `AddOp`.
  Network(const std::string &name);

  // NetDef is the definition of a network, in some occasion, operators are
  // created dynamically by user one by one; but in some other occasion such as
  // LSTM, all the operators in the networks should be  created during the
  // construction of the network. So a `NetDef` is provided with all the
  // operators' definitions.
  Network(const std::string &name, const NetDef &def);

  // Add a operator which is identified  as `type` and has attributes described
  // in `attr`, the `inputs` are the keys of readonly input variables, `outputs`
  // are keys of mutable output variables.
  bool AddOp(const std::string &type, const std::vector<string> &inputs,
             const std::vector<string> &outputs,
             const OprAttr &attr = OprAttr());

  // Add a operator, `Network` will get keys of Variables from `inputs` and
  // `outputs`.
  bool AddOp(const std::string &type,
             const std::vector<const Variable &> &inputs,
             std::vector<Variable &> &outputs, const OprAttr &attr = OprAttr());

  // Run all the operators with the `scope`, if no scope is provided, default
  // scope will be used instead.
  virtual bool Run(Scope *scope = nullptr) override;

  // run all operators in ops_ sequentially.
  bool RunOps(Scope *scope);

private:
  // the operators are owned by `Network`.
  std::vector<std::unique_ptr<Operator>> ops_;
};
```
