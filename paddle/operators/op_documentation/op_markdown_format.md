# Standard Markdown Format for Operators
The following should be the standard format for all the operators that will get rendered in the `html`:

```
Operator Name (In PaddlePaddle)

Operator Name (Standard)

Operator description.

LaTeX equation of how the operator performs an update.

The signature of the operator.
```

The following sections discuss in detail about each of the sections mentioned above.

# Operator Name (In PaddlePaddle)
This should be in all small letters, in case of multiple words, we separate them with an underscore. For example:
array to lod tensor should be written as `array_to_lod_tensor`.

This naming convention should be standard across all PaddlePaddle operators.

# Operator Name (Standard)
This is the standard name of the operator as used in the community. The general standard is as follows:
- All capital letters for standard abbreviations like SGD.
- Use camel-case (capitalize word boundaries inside of a word) for names that have multiple words inside a single word.
- Numbers inside the name kept as is
- Follow the name of the operator with `Activation Operator.`

# Operator description
This section should contain the description of what the operator does, including the operation, the literature from where it came from, where it was introduced first, citing the paper and other important details. The hyperlink of the paper should also be included in this section.

# LaTeX equation
This section should contain an overall equation of the update or operation that the operator performs. The variables used in the equation should follow the naming convention of operators, i.e. two words in the same word should be separated by an underscore (`_`).

# The signature
This section describes the signature of the operator. A list of Inputs and Outputs, each of which have a small description of what the variable represents and the type of variable. The variable names follow the `CamelCase` naming convention.

The following example for an `sgd` operator covers the above mentioned sections as they would look like in the html:

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
