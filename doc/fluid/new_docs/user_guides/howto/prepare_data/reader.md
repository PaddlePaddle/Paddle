```eval_rst
.. _user_guide_reader:
```

# Python Reader

During the training and testing phases, PaddlePaddle programs need to read data. To help the users write code that performs reading input data, we define the following:

- A *reader*: A function that reads data (from file, network, random number generator, etc) and yields the data items.
- A *reader creator*: A function that returns a reader function.
- A *reader decorator*: A function, which takes in one or more readers, and returns a reader.
- A *batch reader*: A function that reads data (from *reader*, file, network, random number generator, etc) and yields a batch of data items.

and also provide a function which can convert a reader to a batch reader, frequently used reader creators and reader decorators.

## Data Reader Interface

*Data reader* doesn't have to be a function that reads and yields data items. It can just be any function without any parameters that creates an iterable (anything can be used in `for x in iterable`) as follows:

```
iterable = data_reader()
```

The item produced from the iterable should be a **single** entry of data and **not** a mini batch. The entry of data could be a single item or a tuple of items. Item should be of one of the [supported types](http://www.paddlepaddle.org/doc/ui/data_provider/pydataprovider2.html?highlight=dense_vector#input-types) (e.g., numpy 1d array of float32, int, list of int etc.)

An example implementation for single item data reader creator is as follows:

```python
def reader_creator_random_image(width, height):
    def reader():
        while True:
            yield numpy.random.uniform(-1, 1, size=width*height)
    return reader
```

An example implementation for multiple item data reader creator is as follows:
```python
def reader_creator_random_image_and_label(width, height, label):
    def reader():
        while True:
            yield numpy.random.uniform(-1, 1, size=width*height), label
    return reader
```

## Batch Reader Interface

*Batch reader* can be any function without any parameters that creates an iterable (anything can be used in `for x in iterable`). The output of the iterable should be a batch (list) of data items. Each item inside the list should be a tuple.

Here are some valid outputs:

```python
# a mini batch of three data items. Each data item consist three columns of data, each of which is 1.
[(1, 1, 1),
(2, 2, 2),
(3, 3, 3)]

# a mini batch of three data items, each data item is a list (single column).
[([1,1,1],),
([2,2,2],),
([3,3,3],)]
```

Please note that each item inside the list must be a tuple, below is an invalid output:
```python
 # wrong, [1,1,1] needs to be inside a tuple: ([1,1,1],).
 # Otherwise it is ambiguous whether [1,1,1] means a single column of data [1, 1, 1],
 # or three columns of data, each of which is 1.
[[1,1,1],
[2,2,2],
[3,3,3]]
```

It is easy to convert from a reader to a batch reader:

```python
mnist_train = paddle.dataset.mnist.train()
mnist_train_batch_reader = paddle.batch(mnist_train, 128)
```

It is also straight forward to create a custom batch reader:

```python
def custom_batch_reader():
    while True:
        batch = []
        for i in xrange(128):
            batch.append((numpy.random.uniform(-1, 1, 28*28),)) # note that it's a tuple being appended.
        yield batch

mnist_random_image_batch_reader = custom_batch_reader
```

## Usage

Following is how we can use the reader with PaddlePaddle:
The batch reader, a mapping from item(s) to data layer, the batch size and the number of total passes will be passed into `paddle.train` as follows:

```python
# two data layer is created:
image_layer = paddle.layer.data("image", ...)
label_layer = paddle.layer.data("label", ...)

# ...
batch_reader = paddle.batch(paddle.dataset.mnist.train(), 128)
paddle.train(batch_reader, {"image":0, "label":1}, 128, 10, ...)
```

## Data Reader Decorator

The *Data reader decorator* takes in a single reader or multiple data readers and returns a new data reader. It is similar to a [python decorator](https://wiki.python.org/moin/PythonDecorators), but it does not use `@` in the syntax.

Since we have a strict interface for data readers (no parameters and return a single data item), a data reader can be used in a flexible way using data reader decorators. Following are a few examples:

### Prefetch Data

Since reading data may take some time and training can not proceed without data, it is generally a good idea to prefetch the data.

Use `paddle.reader.buffered` to prefetch data:

```python
buffered_reader = paddle.reader.buffered(paddle.dataset.mnist.train(), 100)
```

`buffered_reader` will try to buffer (prefetch) `100` data entries.

### Compose Multiple Data Readers

For example, if we want to use a source of real images (say reusing mnist dataset), and a source of random images as input for [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).

We can do the following :

```python
def reader_creator_random_image(width, height):
    def reader():
        while True:
            yield numpy.random.uniform(-1, 1, size=width*height)
    return reader

def reader_creator_bool(t):
    def reader:
        while True:
            yield t
    return reader

true_reader = reader_creator_bool(True)
false_reader = reader_creator_bool(False)

reader = paddle.reader.compose(paddle.dataset.mnist.train(), data_reader_creator_random_image(20, 20), true_reader, false_reader)
# Skipped 1 because paddle.dataset.mnist.train() produces two items per data entry.
# And we don't care about the second item at this time.
paddle.train(paddle.batch(reader, 128), {"true_image":0, "fake_image": 2, "true_label": 3, "false_label": 4}, ...)
```

### Shuffle

Given the shuffle buffer size `n`, `paddle.reader.shuffle` returns a data reader that buffers `n` data entries and shuffles them before a data entry is read.

Example:
```python
reader = paddle.reader.shuffle(paddle.dataset.mnist.train(), 512)
```

## Q & A

### Why does a reader return only a single entry, and not a mini batch?

Returning a single entry makes reusing existing data readers much easier (for example, if an existing reader returns 3 entries instead if a single entry, the training code will be more complicated because it need to handle cases like a batch size 2).

We provide a function: `paddle.batch` to turn (a single entry) reader into a batch reader.

### Why do we need a batch reader, isn't is sufficient to give the reader and batch_size as arguments during training ?

In most of the cases, it would be sufficient to give the reader and batch_size as arguments to the train method. However sometimes the user wants to customize the order of data entries inside a mini batch, or even change the batch size dynamically. For these cases using a batch reader is very efficient and helpful.

### Why use a dictionary instead of a list to provide mapping?

Using a dictionary (`{"image":0, "label":1}`) instead of a list (`["image", "label"]`) gives the advantage that the user can easily reuse the items (e.g., using `{"image_a":0, "image_b":0, "label":1}`) or even skip an item (e.g., using `{"image_a":0, "label":2}`).

### How to create a custom data reader creator ?

```python
def image_reader_creator(image_path, label_path, n):
    def reader():
        f = open(image_path)
        l = open(label_path)
        images = numpy.fromfile(
            f, 'ubyte', count=n * 28 * 28).reshape((n, 28 * 28)).astype('float32')
        images = images / 255.0 * 2.0 - 1.0
        labels = numpy.fromfile(l, 'ubyte', count=n).astype("int")
        for i in xrange(n):
            yield images[i, :], labels[i] # a single entry of data is created each time
        f.close()
        l.close()
    return reader

# images_reader_creator creates a reader
reader = image_reader_creator("/path/to/image_file", "/path/to/label_file", 1024)
paddle.train(paddle.batch(reader, 128), {"image":0, "label":1}, ...)
```

### How is `paddle.train` implemented

An example implementation of paddle.train is:

```python
def train(batch_reader, mapping, batch_size, total_pass):
    for pass_idx in range(total_pass):
        for mini_batch in batch_reader(): # this loop will never end in online learning.
            do_forward_backward(mini_batch, mapping)
```
