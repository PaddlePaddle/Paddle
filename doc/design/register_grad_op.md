# Design Doc: Register Gradient Operator

## Problem

Since we separate users program in two stages, compile time and runtime, we should record and look up the mapping relationship between an operator and its gradient operators when compile. However, we register this relationship in runtime by these `OpInfo` fields.

```cpp
struct OpInfo {
  std::function<OperatorBase*(...)> creator_;
  std::string grad_op_type_;
  ...
};
```

OpInfos store in a association map which key is the operator type. The `grad_op_type` indicate associated gradient operator type. Operator can create gradient operator by `OpInfo::creator_` of gradient. The pseudo code is

```cpp
map<string, OpInfo> OpInfoMap;

OperatorBase* CreateGradientOperator(const OperatorBase& op) {
  return OpInfoMap.at(op.Type()).creator_(...);
}
```

At the same time, an operator's gradient operator could be composed of many forward operators. For example, the gradient operator of `minus_op` could  consist of an `identity` operator and a `scale` operator. To compose a gradient operator by forwarding operators could: 1) Reuse forwarding operator; 2) Calculate second derivative, third derivative, etc.

We use `NetOp` to represent a composed operator since the `NetOp` is `vector<OperatorBase>`. However, `NetOp` is also a runtime concept. We should provide a mechanism to compose operators as a gradient operator.

In conclusion, the problem that we want to resolve in this design doc is to register the mapping relationship between the forward operator and its gradient operators during compile time.


## Solution

The mapping relationship between an operator and its gradient operators is a function. The interface of that function is:

```cpp
// (OpDesc) --> vector<OpDesc>
using GradOpDescMaker = std::function<std::vector<OpDesc>(const OpDesc&)>;
```

The function take a `OpDesc` of the forward operator and return one or many gradient operator descriptions.

The `GradOpDescMaker` will be registered in `OpInfo`, to replace `grad_op_type_` field. The `OpInfo` should be

```cpp
struct OpInfo {
  GradOpDescMaker grad_op_maker_;
  ...
};
```

The `grad_op_maker_ ` is `nullptr` if the operator does not have associated gradient operators.

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
