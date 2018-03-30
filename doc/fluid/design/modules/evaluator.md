# Evaluator Design

## Problem Statement

During training or inference, we provide an evaluation function to measure the model performance, for example, accuracy, precision, etc. In the operator based framework design, the data passes through the network pipeline batch by batch. As a result, inside the operator, we only calculate the metrics for one minibatch. Thus, we need to provide a mechanism to calculate the metrics for each N pass/batch the user wants.

## Evaluator Design
Currently, every operation is expressed in the graph. We divide the evaluator process into three steps.

1. Initialize the metric state and add it into the block.

2. Calculate the concerned metrics for every mini-batch. The single evaluator operator is only responsible for calculating the necessary statistics for one mini-batch. For example, the accuracy operator only calculates the accuracy for a minibatch data if run once.


3. Merge the mini-batch statistics to form the evaluation result for multiple mini-batches. When it comes to distributed training/Multi-GPU training, aggregate the value from different devices.

## Implementation
This design is shown in the Python API.
Each metric operator needs to caculate the metric statistic and return the batch-aware states. Python side is responsible for accumulating the states for each pass.


```python
class Evaluator(object):
    """
    Evaluator Base class.
    """
    def __init__(self, name, **kwargs):
       """
       Different evaluator may has different metric states. E.g, Accuracy need two variables, total and right sample counts.
       Auc need four variables, `true_positives`,
         `true_negatives`, `false_positives` and `false_negatives`. So every evaluator should create its needed variables and append to main_program

       The initialization of Evaluator should be responsible for:
       create metric states and append to the main_program
       """
       pass

    def _update_ops(self, input, label, **kwargs)
       """
       Add mini-batch evaluator caculate operators to the main_program.
       Add increment operator to accumulate the metric states.
       """


    def reset(self, executor, reset_program=None):
      """
      Reset metric states at the begin of each pass/user specified batch number.
      Execute the reset_program to reset the states.
      """


    def eval(self, executor, eval_program=None):
      """
      Merge the mini-batch statistics to form the evaluation result for multiple mini-batches.
      Execute the eval_program and return the result.
      """
      return eval_result
```
