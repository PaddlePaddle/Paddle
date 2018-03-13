# Model CI

The users occasionally found a negligible performance or precision issue between different Paddle versions. Though we have unit tests for each class and Travis-CI to ensures the precision of each operator, there is no any logic to ensure the model (a composition of several operators) works as reliable as the operators.

There are several conditions where an existing model will fail either in performance or precision:

1. the incomplete coverage test cases. 
2. poor performance update, currently, we have no performance tracker for each operator.
3. API changes, developers are likely to forget to update usages in some other repositories such as paddle/models.

The model-CI module is proposed to enhance the weaknesses above and track the overall performance and precision of the model granularity, besides pay attention to the change of python API.

## Make Factor Tracking Extensible

There are some general factors including `train cost`, `validate cost` and `duration`, but some more factors such as `memory usage`, `gpu_memory_usage` and so on should add in the test.

To make it simple and clear to add a new factor into the CI, the framework will offer some interfaces to standardize the functions of factor tracker, such as a base class

```python
class Factor(object):
    dic = {}
    def __init__(self, out_file, his_file=None):
        self.out_file = out_file
        self.his_file = os.path.join('history', out_file)
        self.factors = []
        
    def add_record(self, r): # called when the model run, add execution details.
        self.factors.append(r)
        
    def test(): # called when test
        # can be something comparing the execution details with historical data.
        raise NotImplementedError
        
    @staticmethod
    def register(factor): # factor should be a subclass
        assert isinstance(factor, Factor)
        key = factor.__name__
        assert key not in Factor.dic
        Factor.dic[key] = factor
    
       def __del__(self):
        if self.factors:
            # write to file self.out_file
```

More factors can be integrated into the test framework, for example, a factor tracker which test the training duration can be added in the following way

```python
class TrainDurationFactor(Factor):
    def __init__(self, threshold):
        super(TrainDurationFactor, self).__init__('train.dura.txt')
        self.threshold = threshold
    
    def test(self):
        cur_data = _load_nparray_from_file(self.out_file)
        his_data = _load_nparray_from_file(self.his_file)
        diff = np.abs(cur_data - his_data) / his_data
        if (diff > self.threshold).any():
            raise TestError
# register this factor tracker into the framework      
Factor.register(TrainDurationFactor)
```

A testable model should have a file called `meta.py` with some configurations about those factors to use like

```python
# this is a demo for meta.py
train_duration_factor = TrainDurationFactor(0.1)
valid_duration_factor = ValidDurationFactor()
train_memory_factor = TrainMemoryFactor()

tracking_factors = [ train_duration, valid_duration_factor, train_meta_factor ]
```

Inside the model, one should call the factor trackers and add records.

```python
# configuration of some model
import paddle
import meta
# ...


for batch in batches:
    # ...
    duration = _get_duration_of_this_batch()
    train_duration_factor.add_record(duration)
    # ...
```

and the test framework will test each factor like

```python
for tracker in some_model.meta.tracking_factors:
    some_model._run() # run and tracking_factors will collect running status
    try:
        tracker.test()
    except TestError:
        _collect_error_info
        _ring_alarm
```

## Build a testable model

The models should be placed in `./models` directory, each has a sub-directory, and a `train.xsh` script to define how to run this model. After triggering the `train.xsh`, all the data of `tracking_factors` should be created.

For example, a normal model might have following logic

```python
# train.xsh
run_train_cpu
run_train_gpu
```

To make the testing logic stable, the testable model should assure that

- fix the random seed to make result reproducible
- just run 10 or 20 batches, and each batch takes no more than 30 seconds to limit the overall execution time
- run different modes sequentially, not run them parallelly

## Persistence of log

The log of each execution should be stored somewhere, 
the simplest way is to use Git to maintain a versionable history.

After each execution, add all the logs and statistic result and commit with a comment with a 
template like

```
{success_or_not} {short summary of the cause}

execution duration: {overall_duration}
paddle code version: {commitid}
```

## Alarm

If a test failed, ring an alarm by 

- sending email to `paddle-dev@baidu.com` including
  - error type
  - the details of the tracked abnormal factors
- update the error information and push to git
