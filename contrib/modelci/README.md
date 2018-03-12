# Model CI

A simple Continuous Interation for Models, tracking the overall effect and performance.

## Build testable model
The new models should be located in `./models` directory, and has the following xonsh script files:

- `train.xsh`
- `validate.xsh`

a model can run independently, and support following commands

- `train.xsh --train_cost_out <path> --valid_cost_out <path>`, just train this model on CPU
- `train.xsh --gpu <id> --train_cost_out <path>, --valid_cost_out <path>`, trian this model on a specific gpu.
  - output the errors and time of the training of a model
- `predict.xsh --out_path <path>`, load a model, and predict
  - output the errors and time of the predicting of a model

the log format should like this

for `train_cost` and `valid_cost`, each line is

```
<duration in millisecond>\t<cost>
```

for example:

```
123\t0.98
134\t0.88
```
the first line means this epoch takes 123ms and cost is 0.98.

for prediction, the format is

```
<duration in millisecond>\t<prediction>
```
if the prediction is a vector, use space to concat the elements.

for example, a 3-classification task will output something like

```
45\t0.1 0.4 0.5
```
that means, the epoch takes 45ms and output the probabilities for each class `0.1 0.4 0.5`.
