## Operator Name Convention

To make the operator document itself more clear. we recommend operator names observe the listing conventions.

### Input/Output names

* Variable name is prefer uppercase. e.g. `X`, `Y`. But when the variable is tensor, its name should lowercase. e.g. `matrix`, to discriminate with otherone.

* element wise operator, math operator or similar op, please obey common name convention. if the operator only have one output, use `Out`.

* we prefer more meaningful input/output name. 

### Best Practice
e.g. `rowwise_add`, inputs : `X`, `Y`, outputs : `Out`
e.g. `cosine` , inputs : `X`, `axis`, outputs : `Out`
