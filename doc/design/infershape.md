## InferShape Design

### BackGround
In a DeepLearning framework like PaddlePaddle, a data is a multi-dimension Vector, Operator takes some data as input, do some computation and fulfill other data as output.

Each data has a shape. There are relations between the shape of inputs and outputs for a certain Operator.

Let's take `add` and `mul` as example：

#### 1. `Add_Operator`:

1. the shape of all inputs should be the same.
1. the shape of output should be the same with the input.

#### 2. for `Mul_Operator`:

1. there should be two inputs `X` and `Y`, one output `Out`.
1. `X` and `Y` should be matrix. Assum the shape of `X` is `[x1, x2]`, the shape of `Y` is `[y1, y2]`. Then:
1. x2 must equal to y1.
1. the output shape should be `[x1, y2]`. A user did not need to set the shape of `Out`.

### Why there should have InferShape
From the background above we can see, every Operator will have some similar things to do: check the shape of inputs, set shape of outputs. so we design a uniform interface to do these works.


### What should InferShape do

1. check if the shape of inputs meets the constraint.
1. Set the shape of outputs according to the shape of inputs.

### Interface

PaddlePaddle uses OpMaker to define an Operator. In the OpMaker, User will define the inputs/outputs/attributes of one Operator. For a certain Operator, the things InferShape will do is also certain. So
we design InferShape as a function and register it to OpMaker.

#### The defination of the funciton is:

```cpp
using ShapeInferenceFn =
    std::function<void(const framework::InferShapeContextBase& ctx)>;
```

#### The interface of InferShapeContextBase will be:

```cpp
class InferShapeContextBase {
 public:
  virtual ~InferShapeContextBase() {}
  virtual const framework::DDim get_input_dim(
      const std::string& name) const = 0;
  virtual void set_input_dim(const std::string& name,
                             const framework::DDim& dim) const = 0;
  virtual const framework::DDim get_output_dim(
      const std::string& name) const = 0;
  virtual void set_output_dim(const std::string& name,
                              const DDim& dim) const = 0;
  virtual const AttrReader Attrs() const = 0;
};
```

#### for exmaple
The definition of OpMaker for AddOp will be:

```cpp
class AddOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  AddOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of add op");
    AddInput("Y", "The second input of add op");
    AddOutput("Out", "The output of add op");
    AddComment(R"DOC(
Two Element Add Operator.

The equation is: Out = X + Y
)DOC");

    SetShapeInferenceFn([](const framework::InferShapeContextBase &ctx) {
      PADDLE_ENFORCE_EQ(ctx.get_input_dim("X"), ctx.get_input_dim("Y"),
                        "Two input of Add Op's dimension must be same.");
      ctx.set_output_dim("Out", ctx.get_input_dim("X"));
    });
  }
};
```
