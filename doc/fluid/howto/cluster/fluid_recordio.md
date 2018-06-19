# How to use RecordIO in Fluid

If you want to use RecordIO as your training data format, you need to convert to your training data
to RecordIO files and reading them in the process of training, PaddlePaddle Fluid provides some
interface to deal with the RecordIO files.

## Generate RecordIO File

Before start training with RecordIO files, you need to convert your training data
to RecordIO format by `fluid.recordio_writer.convert_reader_to_recordio_file`, the sample codes
as follows:

```python
    reader = paddle.batch(mnist.train(), batch_size=1)
    feeder = fluid.DataFeeder(
        feed_list=[  # order is image and label
            fluid.layers.data(
            name='image', shape=[784]),
            fluid.layers.data(
            name='label', shape=[1], dtype='int64'),
        ],
        place=fluid.CPUPlace())
    fluid.recordio_writer.convert_reader_to_recordio_file('./mnist.recordio', reader, feeder)
```

The above code snippet would generate a RecordIO `./mnist.recordio` on your host.

**NOTE**: we recommend users to set `batch_size=1` when generating the recordio files so that users can
adjust it flexibly while reading it.

## Use the RecordIO file in a Local Training Job

PaddlePaddle Fluid provides an interface `fluid.layers.io.open_recordio_file` to load your RecordIO file
and then you can use them as a Layer in your network configuration, the sample codes as follows:

```python
    data_file = fluid.layers.io.open_recordio_file(
        filename="./mnist.recordio",
        shapes=[(-1, 784),(-1, 1)],
        lod_levels=[0, 0],
        dtypes=["float32", "int32"])
    data_file = fluid.layers.io.batch(data_file, batch_size=4)

    img, label = fluid.layers.io.read_file(data_file)
    hidden = fluid.layers.fc(input=img, size=100, act='tanh')
    prediction = fluid.layers.fc(input=hidden, size=10, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)

    fluid.optimizer.Adam(learning_rate=1e-3).minimize(avg_loss)

    place = fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    avg_loss_np = []

    # train a pass
    batch_id = 0
    while True:
        tmp, = exe.run(fetch_list=[avg_loss])

        avg_loss_np.append(tmp)
        print(batch_id)
        batch_id += 1
```

## Use the RecordIO files in Distributed Training

1. generate multiple RecordIO files

For a distributed training job, you may have multiple trainer nodes,
and one or more RecordIO files for one trainer node, you can use the interface
`fluid.recordio_writer.convert_reader_to_recordio_files` to convert your training data
into multiple RecordIO files, the sample codes as follows:

```python
    reader = paddle.batch(mnist.train(), batch_size=1)
    feeder = fluid.DataFeeder(
        feed_list=[  # order is image and label
            fluid.layers.data(
            name='image', shape=[784]),
            fluid.layers.data(
            name='label', shape=[1], dtype='int64'),
        ],
        place=fluid.CPUPlace())
    fluid.recordio_writer.convert_reader_to_recordio_files(
          filename_suffix='./mnist.recordio', batch_per_file=100, reader, feeder)
```

The above codes would generate multiple RecordIO files on your host like:

```bash
.
 \_mnist-00000.recordio
 |-mnist-00001.recordio
 |-mnist-00002.recordio
 |-mnist-00003.recordio
 |-mnist-00004.recordio
```

2. open multiple RecordIO files by `fluid.layers.io.open_files`

For a distributed training job, the distributed operator system will schedule trainer process on multiple nodes,
each trainer process reads parts of the whole training data, we usually take the following approach to make the training
data allocated by each trainer process as uniform as possiable:

```python
def gen_train_list(file_pattern, trainers, trainer_id):
   file_list = glob.glob(file_pattern)
   ret_list = []
   for idx, f in enumerate(file_list):
       if (idx + trainers) % trainers == trainer_id:
           ret_list.append(f)
   return ret_list

trainers = int(os.getenv("PADDLE_TRAINERS"))
trainer_id = int(os.getenv("PADDLE_TRAINER_ID"))
data_file = fluid.layers.io.open_files(
    filenames=gen_train_list("./mnist-[0-9]*.recordio", 2, 0),
    thread_num=1,
    shapes=[(-1, 784),(-1, 1)],
    lod_levels=[0, 0],
    dtypes=["float32", "int32"])
img, label = fluid.layers.io.read_file(data_files)
...
```
