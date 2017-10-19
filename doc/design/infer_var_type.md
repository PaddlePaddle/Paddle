# Design Doc: InferVarType

## The Problem Posed

The variable in our design can hold variant types. Such as `LoDTensor` and `SelectedRows`. An operator should be able to inference the variable types of its output.

For example, a `lookup table` operator takes two `LoDTensor`; one is a float tensor as the embedding table, the other is an int tensor as word ID. The gradient operator of `lookup table` will generate a `SelectedRows` as its output. A `sum` operator can take both `LoDTensor` and `SelectedRows` as its inputs and will generate a `LoDTensor` if any of its inputs is `LoDTensor`, otherwise, the `sum` operator will generate `SelectedRows` as its output.

The variable type will be constant at runtime. Every variable's type can either be set by the user (input data and parameter) or be inferred by the operator in compile time.

## Proposed Solution

The `InferVarType` is a compile-time function which is registered to each operator. The inferface of that function is:


```c++
using InferVarTypeFN = std::function<
    void (const OpDescBind& /*op_desc*/, BlockDescBind* /*block*/)>;
```

It takes an operator description as its input and will write the output variable type and store them in block description.

The `InferVarTypeFN` will be registered in `OpInfo`, to replace `infer_var_type_` field. The `OpInfo` should be

```cpp
struct OpInfo {
  InferVarTypeFN infer_var_type_;
  ...
};
```

The default `InferVarType` will set output type as `LoDTensor`. It can be done by `GetInferVarType()`.

```cpp
void DefaultInferVarType(const OpDescBind& op_desc, BlockDescBind* block) {
  // set the output type of variable as `LoDTensor`.
  // ...
}

struct OpInfo {
  InferVarTypeFN infer_var_type_;
  InferVarTypeFN GetInferVarType() const {
    if (infer_var_type_) {
      return infer_var_type_;
    } else {
      return DefaultInferVarType;
    }
  }
};
```

## Register InferVarType

We provide a thin base class for registering an `InferVarTypeFN`. To use a base class will ease the implementation of registry since we can detect the registry entry is an `InferVarTypeFN` or not.

```cpp
class VarTypeInferer {
public:
  virtual void operator()(const OpDescBind& op_desc, BlockDescBind* block) const = 0;
}
```

Operator developers can write the specialize `VarTypeInferer` as follow.

```cpp
class SpecialVarTypeInferer : public VarTypeInferer {
public:
  virtual void operator()(const OpDescBind& op_desc, BlockDescBind* block) const {
    // .. own logic
  }
}
```

Then user can register the `InferVarType` just like `GradOpDescMaker` and `OpInfoMaker`.

```
REGISTER_OPERATOR(some_op, OpType, SpecialVarTypeInferer, ...);
```
