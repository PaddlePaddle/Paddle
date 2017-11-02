## Evaluator Design

### The Problem

During training or serving, we provide the evaluation function to measure the model performance, e.g., accuracy, precision. In the operator based framework design, the data go through the network pipeline batch by batch. As a result, inside the operator, we only can calculate one minibatch metrics. We need to provide a mechanism to calculate the metrics for each N pass/batch the user wanted.

### Evaluator Design
Currently, every operation is expressed in the graph. we divide the evaluator process into three steps.

1. Initialize the metric state necessary and add it into the block.

2. Calculate the statistic of the metric state in every mini-batch. The single operator is only responsible for calculating necessary statistics for one mini-batch. For example, accuracy operator only calculate a minibatch data if run once.\


3. Merge the mini-batch statistics to form the evaluation result for multiple mini-batches. When it comes to distributed training/Multi-GPU training, aggregate the value from different devices.

### Implementation
This design is shown in python API. There would be an abstract python interface and multiple inheritances for each evaluation method.

```python
class Evaluator(object):
    """
    Evalutor Base class.
    """

    def _initialize(self):
    """
    add initialize operators and create metric states to block
    """
        pass

    def _add_evalutor_op(self):
    """
    add mini-batch caculate operators to block
    """
        pass

    def _merge(self);
    """
    Merge the mini-batch statistics to form the evaluation result for multiple mini-batches.
    """
        pass

    def evaluate(self):
    """
    exported interface
    """

```
