# Design Doc: Gradient Operators Registration


## The Problem Posed

Currently, for each C++ operator class definition, a *gradient operator creator* function is registered, which takes as input a C++ operator instance and returns the corresponding gradient operator instance.

However, we noticed two problems with the current design:

1. As we decided to separate the *compilation* and the *execution* phases, we need to change the creator to take an `OpDesc` protobuf message in a `ProgramDesc` and inserts corresponding `OpDesc` messages into the `ProgramDesc` message.

1. For some operators, the gradient computation can be written in terms of existing operators.  For example, the gradient of *minus* operator consists of two operators -- an *identity* operator followed by a *scale* operator.  Hence the registration mechanism needs to support mapping from an operator to a set of operators for the gradient computation.

## The Current Implementation

Instances of the C++ class `OpInfo` are stored an associative map whose key is the operator type. The `grad_op_type` indicates the associated gradient operator type. An operator can create the gradient operator by invoking `OpInfo::creator_` of the gradient operator. The pseudo code is as follows

```cpp
struct OpInfo {
  std::function<OperatorBase*(...)> creator_;
  std::string grad_op_type_;
  ...
};

map<string, OpInfo> OpInfoMap;

OperatorBase* CreateGradientOperator(const OperatorBase& op) {
  return OpInfoMap.at(op.Type()).creator_(...);
}
```

## Proposed Solution

The mapping relationship between an operator and its gradient operators is a function. The interface of this function is:

```cpp
// (OpDesc) --> vector<OpDesc>
std::function<std::vector<OpDescBind>(const OpDescBind&)>;
```

The function takes an `OpDescBind` of the forward operator and returns one or many gradient operator descriptions. `OpDescBind` is a C++ wrapper for  the protobuf message `OpDesc` for rapid manipulation of `OpDesc`.

The `GradOpDescMaker` will be registered in `OpInfo` and will replace the `grad_op_type_` field. The `OpInfo` should look like 

```cpp
struct OpInfo {
  std::function<std::vector<std::unique_ptr<OpDescBind>>(const OpDescBind&)>  grad_op_maker_;
  ...
};
```

The `grad_op_maker_ ` is a `nullptr` if the operator does not have any associated gradient operators.

We propose a base class called `GradOpDescMakerBase` to let operator developers generate `Gradient Operators` easily. The public interface of that class is

```cpp
class GradOpDescMakerBase {
public:
  GradOpDescMakerBase(const OpDescBind& );
  virtual std::vector<std::unique_ptr<OpDescBind>> operator()()const = 0;
};
```

We can convert `GradOpDescMakerBase` to `std::function<std::vector<std::unique_ptr<OpDescBind>>(const OpDescBind&)>` by

```cpp
using GradOpMaker = ...;
std::function<std::vector<OpDescBind>(const OpDescBind&)> func;
func = [] (const OpDescBind& fwd_op) {
  GradOpMaker maker(fwd_op);
  return maker();
};
```

We can write many helper functions since the `GradOpDescMakerBase` is a class now. The basic helper functions get the variables of `Input`, `Output`, `InputGradient` and `OutputGradient` in the forwarding operator.

We should change register macros at the same time. In the current solution, there is no difference between forwarding operators and backward operators. So `REGISTER_OP` just register one operator. If the `REGISTER_OPERATOR ` contains `OpProtoAndCheckerMaker` and `GradOpDescMaker`, we just list them in the same macro. It can be done by a macro contains `__VA_ARGS__`.

The user interface should be

```cpp
vector<OpDesc> MinusOpGradMaker(OpDesc) {...}
REGISTER_OPERATOR(minus, MinusOp, MinusOpProtoAndCheckerMaker, SumOpGradMaker);
// Developers can still manually implement gradient operator.
REGISTER_OPERATOR(minus_grad, MinusGradOp);
```

The interface of current `REGISTER_OP` macro could not be changed. In `REGISTER_OP`, it will invoke `REGISTER_OPERATOR` two times and generate GradOpDescMaker inside.

```cpp
REGISTER_OP(minus, MinusOp, MinusOpProtoAndCheckerMaker, minus_grad, MinusGradOp);
```
