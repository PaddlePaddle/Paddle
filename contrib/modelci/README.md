# Model CI

A simple Continuous Interation for Models, tracking the overall effect and performance.

## Build testable model
The new models should be located in `./models` directory, and has the following xonsh script files:

- `train.xsh`
- `validate.xsh`

a model can run independently, and support following commands

- `train.xsh --train_cost_out <path> --valid_cost_out <path>`, just train this model on CPU
- `train.xsh --gpu <id> --train_cost_out <path>, --valid_cost_out <path>`, train this model on a specific gpu.
  - output the errors and time of the training of a model
- `predict.xsh --out_path <path>`, load a model, and predict
  - output the errors and time of the predicting of a model

the log format should like this

for `train_cost` and `valid_cost`, each line is

```
<duration in millisecond>\t<cost json>
```

for example:

```
123\t[[0.98]]
134\t[[0.88]]
```
the first line means this epoch takes 123ms and cost is a arbitray shape list

for prediction, the format is

```
<duration in millisecond>\t<prediction json>
```

for example, a 3-classification task will output something like

```
45\t[[0.1,0.4,0.5]]
```
that means, the epoch takes 45ms and output the probabilities for each class `0.1 0.4 0.5`.

The validate program will load the JSON format cost, transform it into a numpy array, and 
test the change range of each element.

## make factor tracking extensible

## Persistence of log
The log of each execution should be store somewhere, 
the simplest way is use git to maintain a versionable history.

After each execution, add all the logs and statistic result, and commit with comment with a 
template like

```
{success_or_not} {short summary of the cause}

execution duration: {overall_duration}
paddle code version: {commitid}
```

## Alarm

If a test failed, ring a alarm by 

- sending email to `paddle-dev@baidu.com` including
  - error type
  - the details of the tracked abnormal factors
- update the error infomation and push to git
