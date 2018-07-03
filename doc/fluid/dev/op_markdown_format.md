# Standard Markdown Format for Operators
The following should be the standard format for documentation for all the operators that will get rendered in the `html`:

```
Operator Name (In PaddlePaddle)

Operator Name (Standard)

Operator description.

LaTeX equation of how the operator performs an update.

The signature of the operator.
```

Each section mentioned above has been covered in further detail in the rest of the document.

## PaddlePaddle Operator Name
This should be in all small letters, in case of multiple words, we separate them with an underscore. For example:
`array to lod tensor` should be written as `array_to_lod_tensor`.

This naming convention should be standard across all PaddlePaddle operators.

## Standard Operator Name
This is the standard name of the operator as used in the community. The general standard is usually:
- Standard abbreviations like `SGD` are written in all capital letters.
- Operator names that have multiple words inside a single word use `camelCase` (capitalize word boundaries inside of a word).
- Keep numbers inside a word as is, with no boundary delimiters.
- Follow the name of the operator with the keyword: `Activation Operator.`

## Operator description
This section should contain the description of what the operator does, including the operation performed, the literature from where it comes and was introduced first, and other important details. The relevant paper/article including the hyperlink should be cited in this section.

## LaTeX equation
This section should contain an overall equation of the update or operation that the operator performs. The variables used in the equation should follow the naming convention of operators as described [here](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/name_convention.md). Two words in the same word should be separated by an underscore (`_`).

## The signature
This section describes the signature of the operator. A list of Inputs and Outputs, each of which have a small description of what the variable represents and the type of variable. The variable names follow the `CamelCase` naming convention. The proposed format for this is:
`Section :
VariableName : (VariableType) VariableDescription
...
...
`


The following example for an `sgd` operator covers the above mentioned sections as they would ideally look like in the `html`:

```
sgd

SGD operator

This operator implements one step of the stochastic gradient descent algorithm.

param_out = param_learning_rate * grad

Inputs:
Param : (Tensor) Input parameter
LearningRate : (Tensor) Learning rate of SGD
Grad : (Tensor) Input gradient

Outputs:
ParamOut : (Tensor) Output parameter
```
