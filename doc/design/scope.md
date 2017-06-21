# Scope

### Define

Scope is a context to manage Variables. It mainly contains a map from Variable name to Variable. Net will get and update variable throw scope.

```cpp
class Variable;
using VariablePtr = std::shared_ptr<Variable>;

class Scope final {
 public:
  Scope();
  Scope(const std::shared_ptr<Scope>& parent);

  //! Get Variable in this scope.
  //! @return nullptr if no such variable.
  const VariablePtr& GetVar(const std::string& name) const;

  //! Create or get a variable in this scope.
  VariablePtr& GetOrCreateVar(const std::string& name);

private:
  /// variable name -> variable
  std::unordered_map<std::string, VariablePtr> vars_;
  std::shared_ptr<Scope> parent_{nullptr};
};
```

You need to specify a scope to run a Net. One net can run in different scopes and update different variable in the scope. If you did not specify one, It will run in a default scope.

```cpp
Scope global;
auto x = NewVar("X");  // x is created in scope global, implicitly.
auto y = NewVar("Y");
Net net1;
net1.AddOp("add", {x, y}, {x});  // x = x + y;
net1.Run();

for (size_t i=0; i<10; ++i) {
  Scope local;
  auto tmp = NewVar("tmp");  // tmp is created in scope local.
  Net net2;
  net2.AddOp("add", {x, y}, {tmp});
  net2.Run();  // tmp = x + y;
}

Net net3;
net3.AddOp("add", {x, y}, {"tmp"});  // error! cannot found "tmp" in global scope.

```

### Chain structure

Scope has a pointer point to it's parent scope, this is mainly used in RNN when it need to create many stepNet.


### Scope Guard
