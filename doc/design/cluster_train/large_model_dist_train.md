# Alalysis of large model distributed training in Paddle

***NOTE: This is only some note for how we implemeted this scheme in V1, not a new design.***

## What is it

We often encounter cases that the embedding layer parameters(sparse) are so large that we can not store it in the trainer's memory when training. So we need to put them to several servers, and fetch them row by row instead of fetch all of the parameters.

## How to use

Specify command-line argument like  `--loadsave_parameters_in_pserver=true --ports_num_for_sparse=1  --use_old_updater=1` when starting the paddle trainer. And also add something like `--ports_num_for_sparse=1 --pserver_num_threads=5` when starting pserver processes.

Accrodingly, configure your embedding layers like:

```python
SPARSE_REMOTE=True

w1 = data_layer(name="w1", size=dict_size)
emb1 = embedding_layer(input=w1, size=32, param_attr=ParameterAttribute(sparse_update=SPARSE_REMOTE))
w2 = data_layer(name="w2", size=dict_size)
emb2 = embedding_layer(input=w2, size=32, param_attr=ParameterAttribute(sparse_update=SPARSE_REMOTE))
...
```

## Implementation details

```c++
enum MatType {
  MAT_NORMAL,
  MAT_NORMAL_SHARED,
  MAT_VALUE_SHARED,
  MAT_SPARSE_ROW_IDS,
  MAT_SPARSE_ROW_AUTO_GROW,
  MAT_CACHE_ROW,
  MAT_SPARSE_ROW,
  MAT_SPARSE_ROW_PREFETCH,
  MAT_SPARSE_ROW_PREFETCH_FULL_SIZE,
};
```

`MAT_SPARSE_ROW_PREFETCH` is what we use when configured to fetch only row of matrix when training.

In `trainer_internal.cpp:L93 trainOneBatch`:

```c++
  if (config_->getOptConfig().use_sparse_remote_updater()) {
    REGISTER_TIMER("prefetch");
    gradientMachine_->prefetch(inArgs);
    parameterUpdater_->getParametersRemote();
  }
```

When doing actual network forward and backward, at the beginning of each batch, the trainer will try to download one row of data from pserver.

In `trainer/RemoteParameterUpdater.cpp`: `parameterUpdater_->getParametersRemote();`:

```c++
if (fullSize) {
    ...
} else {
getParams = [&] {
    parameterClient_->getParameterSparse(
        /* recvParameterType= */ PARAMETER_VALUE, sendBackParameterType);
};
applyL1 = [](Parameter& para, real decayRate) {
    para.getMat(PARAMETER_VALUE)->applyL1(/*lr=*/1.0f, decayRate);
};
}
```

Calling `parameterClient_->getParameterSparse` will do remote call to pserver's `getParameterSparse`:

```c++
void ParameterServer2::getParameterSparse(const SendParameterRequest& request,
                                          std::vector<Buffer>& inputBuffers,
                                          SendParameterResponse* response,
                                          std::vector<Buffer>* outputBuffers) {
  (void)inputBuffers;
  auto& buffer = *readWriteBuffer_;
  size_t numReals = 0;
  for (const auto& block : request.blocks()) {
    numReals += getParameterConfig(block).dims(1);
  }
  buffer.resize(numReals);

  VLOG(3) << "pserver: getParameterSparse, numReals=" << numReals;

  ReadLockGuard guard(parameterMutex_);
  size_t offset = 0;
  for (const auto& block : request.blocks()) {
    size_t width = getParameterConfig(block).dims(1);
    Buffer buf = {buffer.data() + offset, width};
    int type = request.send_back_parameter_type();
    sendBackParameterSparse(block, type, response, &buf, width, outputBuffers);
    offset += width;
  }
}
```

`getParameterConfig(block).dims(1)` returns the width of the current "parameter block"(a shard of parameter object),
then `getParameterSparse` remote call returns only one row of data to the client.
