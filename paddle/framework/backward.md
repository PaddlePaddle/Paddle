# Operator/expression 's Backward

## Motivation

In Neural Network, the backpropagation algorithm follows the chain rule, so we need to compound the fundmental gradient operators/expressions together with chain rule . Every forward network need a backward network to construct the full computation lineage, the operator/ expression's Backward feature will generate the backward pass respect to forward pass.
 
## Backward Operator Registry

A backward network is built up with several backward operators. Backward operators take forward operators' inputs, outputs and output gradients, and then calculate its input gradients. In most cases, there is a one-to-one correspondence between forward and backward operators. We use registry mechanism to save these correspondences, which is quite similar with operator registry itself.

For example, we have got a `add_two_op`, and is registered by the following code:

```cpp
REGISTER_OP(add_two, AddTwoOp, AddTwoOpMaker);
```

`add_two` is the operator's type. `AddTwoOp` and `AddTwoOpMaker` are the operator class and the operator maker class respectively.

Assume that we have also got the backward operator of `add_two_op`, which calculating the gradients of `add_two_op`'s inputs. Then we register it by the following way:

```cpp
REGISTER_GRADIENT_OP(add_two, add_two_grad, AddTwoGradOp);
```

`add_two_grad` is the type of backward operator, and `AddTwoGradOp` is its class name.

### Implement : gradient operator registry

|                        | forward operator | backward operator                |
| ---------------------- | ---------------- | -------------------------------- |
| **Operator::inputs_**  | Inputs           | Inputs, Outputs, OutputGradients |
| **Operator::outputs_** | Outputs          | InputGradients                   |

Inputs/Outputs means the input/output of the operator,  InputGradients/OutputGradients is the gradient respect to forward opeartor. Forward operator and Backward operator are isomorphic, save their corresponding needs into member attribute.

We use a global hash map record the gradient operators available, follow the philosophy  of minimum core, make operator pluggable unit. Each gradient is an operator and it needs to regist itself. 

grad_op_builder(fengjiayi)

### Implement : Backward network

given a forward network, it generates the backward network. We only care about the Gradients—`OutputGradients`,`InputGradients`.

1. bla bla bla (yuyang)

2. NetOp 

   when the input forward network is a NetOp, it need to call the sub NetOp/Operators backward function recursively and ensure them done. During the process, we need to collect the `OutputGradients` name.

   We share variable in the same scope, as a result, duplicate operator `OutputGradients` will overwirte then duplicate variable.  

   ![./images/duplicate_op]()

    Share variable between operators or same input variable used in multiple operators lead to a duplicate gradient variable. As demo show above, we need to rename gradient name recursively, and add a generic add operator instead. 

![./images/duplicate_op2]()

​	Then collect the sub graph OutputGradients/InputGradients as the NetOp's and return it.
