# Dynamic Graph on Fluid

We can implement Dynet-like Tape(See this survey) by wrapping Paddle Fluid's `Operator`
and `Variable`.

The user API is straight forward since

1. it is imperative. And it uses host language's control flow logic.
1. it avoids extra concepts such as `Scope` and `Executor`.

All of these benefits come at the cost of just adding one line `reset_global_tape`
at every iteration.

## Code Structure

In short, the `Tape` contains a vector of `OpHandle`s. And an `OpHandle` contains its
`type`, the pointers to the `Variable`s, and necessary attributes.

```c++
class Variable {
public:
  VriableHandle Grad(); // returns its gradient variable
private:
  framework::VarDesc desc_; // compile time infershape, necessary for lazy execution
  framework::Variable var_; // run time variable, holds data memory
};

using VariableHandle = shared_ptr<Variable>;

struct OpHandle {
  string type_;
  map<string, vector<VariableHandle>> inputs_;
  map<string, vector<VariableHandle>> outputs_;
  AttributeMap attrs_;
};

class Tape {
public:
  void AddOp(OpHandle); // add op
  void Forward();       // execute the tape_
  void Backward();      // execute the backward of the tape_
private:
  vector<OpHandle> tape_;
};
```

We uses `Function` to indicate layers. It takes care of parameter
initialization and `AddOp` to the Tape when it is called.

```c++
class Linear {
 public:
  Linear(int in_dim, int out_dim, const std::string &act)
      : w_(new Variable("LinearWeight")),
        b_(new Variable("LinearBias")),
        act_(act) {
    Tape init_tape;

    std::string initializer = "fill_constant";
    framework::AttributeMap attrs;
    attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["shape"] = std::vector<int>{in_dim, out_dim};
    attrs["value"] = 1.0f;
    init_tape.AddOp(initializer, {}, {{"Out", {w_}}}, attrs);

    attrs["dtype"] = paddle::framework::proto::VarType::Type::VarType_Type_FP32;
    attrs["shape"] = std::vector<int>{out_dim};
    attrs["value"] = 1.0f;
    init_tape.AddOp(initializer, {}, {{"Out", {b_}}}, attrs);

    init_tape.Forward();
  }

  VariableHandle operator()(VariableHandle input) {
    VariableHandle pre_bias(new Variable("linear"));
    get_global_tape().AddOp("mul",
                            {{"X", {input}}, {"Y", {w_}}},
                            {{"Out", {pre_bias}}},
                            {{"x_num_col_dims", 1}, {"y_num_col_dims", 1}});
    VariableHandle pre_act(new Variable("linear"));
    get_global_tape().AddOp("elementwise_add",
                            {{"X", {pre_bias}}, {"Y", {b_}}},
                            {{"Out", {pre_act}}},
                            {{"axis", 1}});
    VariableHandle post_act(new Variable("linear"));
    get_global_tape().AddOp(act_,
                            {{"X", {pre_act}}},
                            {{"Out", {post_act}}},
                            {});
    return post_act;
  }

  std::vector<VariableHandle> Params() { return {w_, b_}; }

 private:
  VariableHandle w_;
  VariableHandle b_;
  std::string act_;
};
```

## User API

```c++
// Model function
paddle::tape::Linear linear1(3, 3, "relu"); // init weight and bias
paddle::tape::Linear linear2(3, 3, "relu"); // init weight and bias
paddle::tape::Mean mean;

// Optimizer
paddle::tape::SGD sgd(0.001);

// Data Feeder
paddle::tape::Fill data_feeder(...);
VariableHandle input(new paddle::tape::Variable("input"));

for (int i = 0; i < 2; ++i) {
  reset_global_tape();

  data_feeder(input);

  auto loss = mean(linear2(linear1(input))); // compile time InferShape & InferVarType
  LOG(INFO) << loss.value(); // Run forward up to loss

  // Run backward, store gradient of w at w->Grad()
  get_global_tape.Backward(loss);

  // Update w
  sgd(linear1.Params());
  sgd(linear2.Params());
}
```

## Code Reuse

We want to stay close to Paddle Fluid as much as possible.

### Reuse All Operators

As all Ops are registered at `OpInfoMap`, the effort of adding a new `Function`
is about 10 lines of code, similar to expose an operator to Python.

### Reuse Compile Time InferShape and InferVarType

Note that all the symbolic information is stored at `tape::Varaible::desc_`, instead
of `ProgramDesc.block.vars`, we create a temporary `BlockDesc` to do `InferShape` and
`InferVarType` every time we `AddOp` to the tape.

### Reuse Operator::Run

We use smart pointer, instead of `Scope`, to manage memory. So we create a temporary
`Scope` for every `Operator::Run()`.

## Possible Feature

### Release Memory on Backward

We can release memory aggressively. During backward, we can delete the OpHandle once
we have finished its backward. Since all the variable is managed by smart pointer, the
memory is automatically released when its `ref_count` goes to 0.

### Kernel Fusion

As a symbolic representation of the Tape is constructed first before the actual
execution, it would be possible to perform graph optimization. One use case is kernel
fusion.
