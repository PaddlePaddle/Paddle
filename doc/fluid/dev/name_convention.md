# Operator's Parameter Name Convention

To make the operator document itself more clear, we recommend operator names obey the listing conventions.

## OpProtoMaker names

When defining an operator in Paddle, a corresponding [OpProtoMaker](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/operator.h#L170) (TODO: OpProtoMaker Doc)need to be defined. All the Input/Output and Attributes will write into the [OpProto](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/framework/framework.proto#L61) , and will be used in client language to create operator.

- Input/Output.
  - Input/Output names follow the **CamelCase**. e.g. `X`, `Y`, `Matrix`, `LastAxisInMatrix`. Input/Output much more like Variables, we prefer to meaningful English words.
  - If an operator's Input/Output are tensors in math, not match to any meaningful words, input name should starts from `X`. e.g. `X`, `Y`, and output name should starts from `Out`. e.g. `Out`. This rule intends making operators which have few inputs/outputs unified.

- Attribute.
  - Attribute name follows the **snake_case**. e.g. `x`, `y`, `axis`, `rowwise_matrix`. Also, attribute name prefers to meaningful English words.

- Comments.
  - Input/Output/Attr comment follow the format of **(type,default value) usage**, corresponding to which type it can be and how it will be used in the operator. e.g.  Attribute in Accumulator`"gamma" `,`(float, default 1.0) Accumulation multiplier`.
  - Operator comment format of` R"DOC(your comment here)DOC"`. You should explain the input/output of the operator first. If there is math calculation in this operator, you should write the equation in the comment. e.g. `Out = X + Y`.

- Order.
  - Follow the order of Input/Output, then Attribute, then Comments. See the example in best practice.

## Best Practice

Here we give some examples to show how these rules will be used.

- The operator has one input, one output. e.g.`relu`, inputs: `X`, outputs: `Out`.

- The operator has two input, one output. e.g. `rowwise_add`, inputs : `X`, `Y`, outputs : `Out`.

- The operator contains attribute. e.g. `cosine`, inputs : `X`, `axis`, outputs : `Out`.

  We give a full example of Accumulator Operator.

```c++
class AccumulateOpMaker : public framework::OpProtoAndCheckerMaker {
public:
  AccumulateOpMaker(OpProto *proto,
                    OpAttrChecker *op_checker)
    : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) The input tensor that has to be accumulated to the output tensor.
    If the output size is not the same as input size,
    the output tensor is first reshaped and initialized to zero, and only then, accumulation is done.");
    AddOutput("Out", "(Tensor) Accumulated output tensor");
    AddAttr<float>("gamma", "(float, default 1.0) Accumulation multiplier").SetDefault(1.0f);
    AddComment(R"DOC(
Accumulate Operator.

This operator accumulates the input tensor to the output tensor. If the
output tensor already has the right size, we add to it; otherwise, we first
initialize the output tensor to all zeros, and then do accumulation. Any
further calls to the operator, given that no one else fiddles with the output
in the interim, will do simple accumulations.

Accumulation is done as follows:

Out = 1*X + gamma*Out

where X is the input tensor, Out is the output tensor and gamma is the multiplier
argument.

)DOC");
  }
};
```
