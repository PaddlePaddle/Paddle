# Operator/expression 's Backward

## Motivation

In Neural Network, the backpropagation algorithm follows the chain rule, so we need to compound the fundmental gradient operators/expressions together with chain rule . Every forward network need a backward network to construct the full computation lineage, the operator/ expression's Backward feature will generate the backward pass respect to forward pass.
 
## Backward Operator Registry

A backward network is built up with several backward operators. Backward operators take forward operators' inputs, outputs and output gradients and then calculate its input gradients. In most cases, there is a one-to-one correspondence between forward and backward operators. We use registry mechanism to save these correspondences, which is quite similar with operator registry itself.

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

## Backward Opeartor Creating

### Usage

Given a certain forward operator, we can get its corresponding backward opeartor by calling:

```cpp
OperatorBase* bwd_op = BuildGradOp(const OperatorBase* fwd_op);
``` 

The function `BuildGradOp` will sequentially execute following processes:

1. Getting the `type_` of given forward operator, and then creating the corresponding backward operator.

2. Copying all the attributes of forward operator expect `input_format` and `output_format`(if it has), for their elements differ between forward and backward operators.

3. Copying forward operator's `inputs_` and `outputs_` to backward operator's `inputs_`. And adding forward inputs' gradient variables into backward `output_`, adding forward outputs' gradient variables into backward `input_`.

4. Building backward operator's `input_format`, `output_format` (if necessary) and `in_out_idxs_` according to its `inputs_` and `outputs_` just created.

## Backward Network Building

A backward network is a series of backward operators. The main idea of building a backward network is creating backward operators in the inverted sequence and put them together.

In our design, the network itself is also a kind of operator. So the operators contained by a big network may be some small network. 

given a forward network, it generates the backward network. We only care about the Gradients—`OutputGradients`,`InputGradients`.

1. bla bla bla (yuyang)

2. NetOp 

   when the input forward network is a NetOp, it need to call the sub NetOp/Operators backward function recursively and ensure them done. During the process, we need to collect the `OutputGradients` name.

   We share variable in the same scope, as a result, duplicate operator `OutputGradients` will overwirte then duplicate variable.  

   ![./images/duplicate_op]()

    Share variable between operators or same input variable used in multiple operators lead to a duplicate gradient variable. As demo show above, we need to rename gradient name recursively, and add a generic add operator instead. 

![./images/duplicate_op2]()

​	Then collect the sub graph OutputGradients/InputGradients as the NetOp's and return it.
