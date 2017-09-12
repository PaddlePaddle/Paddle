## Operator Name Convention

To make the operator document itself more clear, we recommend operator names obey the listing conventions.

### OpMaker names

When defining an operator in Paddle, a corresponding `OpMaker` need to be defined. All the `Input`/`Output` and `attrs` will write into the `OpProto` , and will be used in client language to create operator. 

- Input/Output.
  -  names follow the `CamelCase` but the first character is uppercase. e.g. `X`, `Y`, `Matrix`, `LastAxisInMatrix`. Input/Output much more like Variables, we prefer to meaningful English words.
  - If an operator's Input/Output are not meaningful words, input name starts from `X`. e.g. `X`, `Y`, and output name starts from `Out`. e.g. `Out`.

* Attribute.
  * Attribute name follows the normal `CamelCase`. e.g. `x`, `y`, `axis`, `rowwiseMatrix`. Also, attribute name prefers to meaningful English words.
* Comments.
  * Input/Output/Attr comment follow the format of `type:meaning`. e.g. `AddOutput("Out", "EigenTensor,Tensor: Output of XX")`. we prefer to more meaningful comment. Some comments like `The first input of Operator` contains no information, we forbid it.
  * Operator comment format of` R"DOC(your comment here)DOC"`. if there is math calculation in this operator, you should write the equation in the comment. e.g. `Out = X + Y`. 

### Best Practice

- The operator has one input, one output. e.g.`relu`, inputs: `X`, outputs: `Out`. 

- The operator has two input, one output. e.g. `rowwise_add`, inputs : `X`, `Y`, outputs : `Out`.

- The operator contains attribute. e.g. `cosine`, inputs : `X`, `axis`, outputs : `Out`.

  ​
