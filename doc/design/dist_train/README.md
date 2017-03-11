# Distributed Training Design Doc

## Goal

### Easy To Use

- Switching from local training to distributed training should be seamless.
- Researcher can do it effortlessly without the help from engineers.

## Dataset On Cloud

Since distributed training happens on the cloud, dataset needs to live on the cloud as well. We will discuss how to use and create a dataset for distributed training.

### Predefined Dataset

Any dataset in `paddle.dataset` package is a predefined dataset. User can get the [**data reader**](../reader) from the dataset:

```python
reader = paddle.dataset.cifar.train100()
```

Decorate the `reader` as usual:

```python
buffered = paddle.reader.buffered(reader, 200)
batch = paddle.batch(buffered, 128)
```

And do distributed train:

```python
paddle.dist_train(cost, batch, ...)
```

Or do local train:

```python
paddle.train(cost, batch, ...)
```

### Custom Dataset

A custom dataset is identified by name. User only needs to know the dataset name to train on the dataset.
The complexities like where the dataset is stored, how the dataset is split and distributed are hidden.

User will get a data reader when referencing the dataset by name. User can still decorate data reader as usual, but different from the predefined dataset, the final data reader can only be used for `paddle.dist.train`.

```python
reader = paddle.dist.dataset("imagenet-custom")
buffered = paddle.reader.buffered(reader, 200)
batch = paddle.batch(buffered, 128)
paddle.dist.train(cost, batch, ...)
```

### Create Custom Dataset

User needs to implement data reader that outputs any type in the following list:

```text
bool, bool list, int, int list, float, float list, str list, str (binary data is represented by str)
```

Still, reader can output multiple results each time (e.g., `yield a, b, c`).

Note that numpy array is not in this list, this is for compability with other languages. User can easily convert numpy array into `float list`.

And then user can pass the reader to `paddle.dist.create_dataset(name, reader)` to create the dataset.

Under the hood,

- when `paddle.dist.create_dataset` is run locally, it will [package](#packaging) the program. And run the packaged program on the cloud.

- when `paddle.dist.create_dataset` is run on the cloud, it will save the output of reader into distributed storage and mark them with corresponding dataset name. PaddlePaddle cloud will decide what format the data instances will be saved. The data format will be optimized for sequential read because the [data reader interface](../reader#data-reader-interface) only support sequential read.

Because entire directory where the python file lives in will be packaged. When creating reader, user can reference files in the same directory of python file by relative path, download file from network, or synthesize data.

## Distributed Training

`paddle.dist.train` will do distributed training. It will take reader:

- created from `paddle.dataset` package, or
- created from `paddle.dist.dataset`.

Under the hood,

- when `paddle.dist.traian` is run locally, it will [package](#packaging) the program. And run the packaged program on the cloud.

- when `paddle.dist.train` is run on the cloud, reader will output the sharded training data. The sharded training data is scheduled by training master. Parameters will be synced with parameter servers.

## Packaging

We need to tell cluster what graph to train, how to train, and what data to train on.

Theoretically, we can encode all these three pieces of information into a computing graph and implementing packaging will be simple: Just send the serialized graph to the cloud, and run the graph there.

However, this approach forces user to learn how to do everything using graph operation, and abandon the way user already familiar with using their native programming language.

For example:

- User needs to learn how to queue data using a graph operation when she is already familiar with queue in python.
- User needs to learn how to decode a jpeg file using a graph operation when she already knows how to do it in python.

We think this approach conflict with our goal of easy to use. Instead, we propose the following way of packaging:

The directory of where the main python file resides is the working directory. Base docker image will be PaddlePaddle's official image.

- User will create `requirement.txt` under working directory for any third party python dependencies.
- If user wants additional environment configuration, she can create `build.sh` under working directory.

Docker image will be built by:

- The entire working directory will be copied into the image.
- `pip install -r requirements.txt` will be run.
- `build.sh` will be run.
- The main python file will be configured to run when the docker image runs.

At last, the newly created image will be uploaded to cloud.
