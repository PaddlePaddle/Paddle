# Scope

### Define

Scope is a context to manage Variables. It mainly contains a map from Variable name to Variable. Net will get and update variable throw scope.

```cpp
class Scope {
    Variable GetVar();

private:
    // var_name -> var
    std::map<string, Variable> var_map_;
    Scope* parent_scope_;
}
```

You need to specify a scope to run a Net. One net can run in different scopes and update different variable in the scope. If you did not specify one, It will run in a default scope.
```python
with ScopeGuard(scope)：
    Net net = Net();
    Net.run()
```

### Chain structure

Scope has a pointer point to it's parent scope, this is mainly used in RNN when it need to create many stepNet.


### Scope Guard