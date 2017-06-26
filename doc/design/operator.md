# Operator Design

Operator in PaddlePaddle mainly describe how to do operation with Variable. It does not actually contains any data or state, but with reference of these Variables/State.

Op will get/update data/state from Scope when running.

Operator have a template parameter `Context`. Context is used to specify on which device this op will run. Each Op will implement multi Op according to different Context.

```cpp
#pragma once

namespace paddle {
namespace framework {

class OperatorBase {
public:
    explicit OperatorBase(const OperatorDesc& desc);
    virtual ~OperatorBase() {}
    virtual Error Run(Scope* scope, Context* context) const = 0;

private:
    std::string type_;
    std::vector<const std::string> inputs_;
    std::vector<const std::string> outputs_;
    std::map<std::string, AttrDef* Attr> attrs_;
};

// Operator is the class your should derive when implement a new Operator.
template <typename T>
class Operator : public OperatorBase {
public:
    explicit Operator(const OperatorDesc& desc): OperatorBase(desc) {}

    // This function will
    Error Run(Scope* scope, T* context) const final {}

    // when implement an Op, your should implement this function.
    virtual Error Run(std::vector<Variable> inputs,
                      std::vector<Variable> outputs,
                      T* context) const = 0;
private:
};

}  // namespace framework
}  // namespace paddle
```

