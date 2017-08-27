## Operator/expression 's Backward

### Motivation

In Neural Network, the backpropagation algorithm follows the chain rule, so we need to compound the fundmental gradient operators/expressions together with chain rule . Every forward network need a backward network to construct the full computation lineage, the operator/ expression's Backward feature will generate the backward pass respect to forward pass.

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

1. Op 

   when the input forward network is a Op, return its gradient Operator Immediately.

2. NetOp 

   when the input forward network is a NetOp, it need to call the sub NetOp/Operators backward function recursively. During the process, we need to collect the `OutputGradients` name according to forward NetOp.

   **shared variable**. As illustrated in the pictures, two operator's `Output` `Gradient` will overwirte their shared input variable.  

   <p align="center">
   <img src="./images/duplicate_op.png" width="70%" ><br/>

   1. shared variable in two operators. 

   </p>

   Share variable between operators or same input variable used in multiple operators lead to a duplicate gradient variable. As demo show above, we need to rename gradient name recursively, and add a generic add operator replace the overwirte links. 

   <p align="center">
   <img src="images/duplicate_op2.png" width="90%" ><br/>

   2. replace shared variable gradient with `Add` Operator

   </p>



​	Then collect the sub graph `OutputGradients`/`InputGradients` as the NetOp's and return it.
