# Operator Design

`Operator` in PaddlePaddle mainly describes how to do an operation with `Variable`. It does not actually contain any data or state, but with the key to find these `Variables/State` from `Scope`.

Op will get/update data/state from Scope when running. It should not change the information inside op so all it's `Run()` function should be `const`

`Operator` has a template parameter `DeviceContext`. DeviceContext is used to specify on which device this op will run. Each Op will implement multi Op according to different Context.

#### We design the Operator class with tree layer.

1. OperatorBase
1. Operator
1. CustomOperator


#### `OperatorBase`

`OperatorBase` is the base class of an Operator, but without any specialized information. Because NetworkBase can treat all Ops as OperatorBase and do not need to consider the type or context of these ops. It can just call `Run()` of Op:

```cpp
class NetworkBase {
public:
    Error Run(Scope* scope, Context* context) {
      for (auto& op : operators_) {
        Error err = op->Run(scope, context);
        if (!err.isOk()) {
          return err;
        }
      }
      return Error();
    }

private:
    vector<unique_ptr<OperatorBase>> operators_;
};
```

#### `Operator`

`Operator` is the operator with DeviceContext information. It's the middle layer that handles the context converting and data prefetching work.

It has two Run() function.

1. `Error Run(Scope* scope, Context* context) const final`

This Run() is derived from OperatorBase, It's used to convert BaseContext to CPUContext or GPUContext and get Variables from scope and then passed them to the next Run()

2. `Error Run(std::vector<Variable*>& inputs, std::vector<Variable*>& outputs, DeviceContext* context) const overrid`

This Run() should be derived by Customer Operator class and put there calculate logic in the function. It has all the Variable and Context with right type to run.

#### `CustomOperator`

`CustomOperator` is the operator that should be implemented such as FcLayer. They are derived from the second `Operator` with certain `DeviceContext`, they know where they are running and how to do the right operation.


The following is the pseudocode for these three layers.


```cpp
#pragma once

namespace paddle {
namespace framework {

// OperatorBase provide base element of an Operator without any template.
class OperatorBase {
public:
    explicit OperatorBase(const OpDesc& desc);
    virtual ~OperatorBase() {}

    // initialize Attributes of this OP from proto message desc.attrs()
    // you should derive this function to init the attr you need in OP.
    virtual Error InitializeAttributes(const AttrbuteMap& attrs) = 0;
    virtual Error Run(Scope* scope, Context* context) const = 0;

private:
    std::string type_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
};

// Operator is the class your should derive when implement a new Operator.
template <typename DeviceContext>
class Operator : public OperatorBase {
public:
    explicit Operator(const OpDesc& desc): OperatorBase(desc) {}

    // This function will get all input and output Vars from scope and ten call
    // Run(std::vector<Variable> inputs, std::vector<Variable> outputs, T* context)
    Error Run(Scope* scope, Context* context) const final {
      DeviceContext* dev_context = dynamic_cast<DeviceContext*>(context);
      if (dev_context == nullptr) {
        return Error("dynamic_cast devContext failed!");
      }

      std::vector<Variable*> input_vars;
      std::vector<Variable*> output_vars;

      input_vars.reserve(inputs_.size());
      for(auto& input: inputs_) {
        input_vars.push_back(scope->getOrCreateVariable(input));
      }
      output_vars.reserve(outputs_.size());
      for(auto& input: outputs_) {
        output_vars.push_back(scope->getOrCreateVariable(input));
      }

      return Run(input_vars, output_vars, dev_context);
    }

    // when implement an Op, your should implement this function.
    virtual Error Run(std::vector<Variable*>& inputs,
                      std::vector<Variable*>& outputs,
                      DeviceContext* context) const = 0;
};


// Sample Operator implement. Show how to implement a Cosine Operator.
template <typename DeviceContext>
class CosineOp final : public Operator<DeviceContext> {
public:
    CosineOp(const OpDesc& desc):
            Operator<DeviceContext>(desc) {};

    // init attrs that are needed by this Operator, check the legality here.
    Error InitializeAttributes(const AttrbuteMap& attrs) {
      scale_ = attrs.get<float>("scale");
      if (scale_ <= 0.0) {
        return Error("scale of CosineOp must be larger than 0.0, get %f", scale_);
      }
      return Error();
    }

    // Add the actual calculate logic in this function.
    Error Run(std::vector<Variable*>& inputs,
              std::vector<Variable*>& outputs,
              DeviceContext* context) const override {
      // TODO(to be implement)
      return Error();
    }

private:
    float scale_;
};

}  // namespace framework
}  // namespace paddle
```
