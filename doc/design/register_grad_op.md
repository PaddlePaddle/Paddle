# Design Doc: Gradient Operators Registration


## The Problem Posed

Currently, for each C++ operator class definition, there registers a *gradient operator creator* function, which takes a C++ operator instance and returns the corresponding gradient operator instance.

However, we noticed two problems with the current deisgn:

1. As we decided to separate the *compilation* and *execution* phases, we need to change the creator to take an `OpDesc` protobuf message in a `ProgramDesc` and inserts corresponding `OpDesc` messages into the `ProgramDesc` message.

1. Some operator's gradient computation requires more than one gradient operators.  For example, the gradient of *minus* consists of two operators -- an identity operaotr and a scale operator.  So we need to make the registration mechanism to support the mapping from an operator to a set of operators for gradient computation.

## The Current Implementation

The C++ class `OpInfos` store in a association map which key is the operator type. The `grad_op_type` indicate associated gradient operator type. Operator can create gradient operator by `OpInfo::creator_` of gradient. The pseudo code is

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

The mapping relationship between an operator and its gradient operators is a function. The interface of that function is:

```cpp
// (OpDesc) --> vector<OpDesc>
std::function<std::vector<OpDescBind>(const OpDescBind&)>;
```

The function takes an `OpDescBind` of the forward operator and returns one or many gradient operator descriptions. `OpDescBind` is a C++ wrapper for protobuf message `OpDesc` to manipulate `OpDesc` fast.

The `GradOpDescMaker` will be registered in `OpInfo`, to replace `grad_op_type_` field. The `OpInfo` should be

```cpp
struct OpInfo {
  std::function<std::vector<std::unique_ptr<OpDescBind>>(const OpDescBind&)>  grad_op_maker_;
  ...
};
```

The `grad_op_maker_ ` is `nullptr` if the operator does not have associated gradient operators.

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

We should chagne register macros at the same time. In the current solution, there is no difference between forwarding operators and backward operators. So `REGISTER_OP` just register one operator. If the `REGISTER_OPERATOR ` contains `OpProtoAndCheckerMaker` and `GradOpDescMaker`, we just list them in the same macro. It can be done by a macro contains `__VA_ARGS__`.

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
