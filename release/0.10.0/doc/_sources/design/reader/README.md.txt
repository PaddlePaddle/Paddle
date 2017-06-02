# Python Data Reader Design Doc

At training and testing time, PaddlePaddle programs need to read data. To ease the users' work to write data reading code, we define that

- A *reader* is a function that reads data (from file, network, random number generator, etc) and yields data items.
- A *reader creator* is a function that returns a reader function.
- A *reader decorator* is a function, which accepts one or more readers, and returns a reader.
- A *batch reader* is a function that reads data (from *reader*, file, network, random number generator, etc) and yields a batch of data items.

and provide function which converts reader to batch reader, frequently used reader creators and reader decorators.

## Data Reader Interface

Indeed, *data reader* doesn't have to be a function that reads and yields data items. It can be any function with no parameter that creates a iterable (anything can be used in `for x in iterable`):

```
iterable = data_reader()
```

Element produced from the iterable should be a **single** entry of data, **not** a mini batch. That entry of data could be a single item, or a tuple of items. Item should be of [supported type](http://www.paddlepaddle.org/doc/ui/data_provider/pydataprovider2.html?highlight=dense_vector#input-types) (e.g., numpy 1d array of float32, int, list of int)

An example implementation for single item data reader creator:

```python
def reader_creator_random_image(width, height):
    def reader():
        while True:
            yield numpy.random.uniform(-1, 1, size=width*height)
    return reader
```

An example implementation for multiple item data reader creator:
```python
def reader_creator_random_image_and_label(width, height, label):
    def reader():
        while True:
            yield numpy.random.uniform(-1, 1, size=width*height), label
    return reader
```

## Batch Reader Interface

*batch reader* can be any function with no parameter that creates a iterable (anything can be used in `for x in iterable`). The output of the iterable should be a batch (list) of data items. Each item inside the list must be a tuple.

Here are valid outputs:
```python
# a mini batch of three data items. Each data item consist three columns of data, each of which is 1.
[(1, 1, 1),
(2, 2, 2),
(3, 3, 3)]

# a mini batch of three data items, each data item is a list (single column).
[([1,1,1],),
([2,2,2],),
([3,3,3],),
```

Please note that each item inside the list must be a tuple, below is an invalid output:
```python
 # wrong, [1,1,1] needs to be inside a tuple: ([1,1,1],).
 # Otherwise it's ambiguous whether [1,1,1] means a single column of data [1, 1, 1],
 # or three column of datas, each of which is 1.
[[1,1,1],
[2,2,2],
[3,3,3]]
```

It's easy to convert from reader to batch reader:
```python
mnist_train = paddle.dataset.mnist.train()
mnist_train_batch_reader = paddle.batch(mnist_train, 128)
```

Also easy to create custom batch reader:
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

batch reader, mapping from item(s) read to data layer, batch size and number of total pass will be passed into `paddle.train`:

```python
# two data layer is created:
image_layer = paddle.layer.data("image", ...)
label_layer = paddle.layer.data("label", ...)

# ...
batch_reader = paddle.batch(paddle.dataset.mnist.train(), 128)
paddle.train(batch_reader, {"image":0, "label":1}, 128, 10, ...)
```

## Data Reader Decorator

*Data reader decorator* takes a single or multiple data reader, returns a new data reader. It is similar to a [python decorator](https://wiki.python.org/moin/PythonDecorators), but it does not use `@` syntax.

Since we have a strict interface for data readers (no parameter, return a single data item). Data reader can be used flexiable via data reader decorators. Following are a few examples:

### Prefetch Data

Since reading data may take time and training can not proceed without data. It is generally a good idea to prefetch data.

Use `paddle.reader.buffered` to prefetch data:

```python
buffered_reader = paddle.reader.buffered(paddle.dataset.mnist.train(), 100)
```

`buffered_reader` will try to buffer (prefetch) `100` data entries.

### Compose Multiple Data Readers

For example, we want to use a source of real images (reusing mnist dataset), and a source of random images as input for [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).

We can do:

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
# And we don't care second item at this time.
paddle.train(paddle.batch(reader, 128), {"true_image":0, "fake_image": 2, "true_label": 3, "false_label": 4}, ...)
```

### Shuffle

Given shuffle buffer size `n`, `paddle.reader.shuffle` will return a data reader that buffers `n` data entries and shuffle them before a data entry is read.

Example:
```python
reader = paddle.reader.shuffle(paddle.dataset.mnist.train(), 512)
```

## Q & A

### Why reader return only a single entry, but not a mini batch?

Always returning a single entry make reusing existing data readers much easier (e.g., if existing reader return not a single entry but 3 entries, training code will be more complex because it need to handle cases like batch size 2).

We provide function `paddle.batch` to turn (single entry) reader into batch reader.

### Why do we need batch reader, isn't train take reader and batch_size as arguments sufficient?

In most of the case, train taking reader and batch_size as arguments would be sufficent. However sometimes user want to customize order of data entries inside a mini batch. Or even change batch size dynamically.

### Why use a dictionary but not a list to provide mapping?

We decided to use dictionary (`{"image":0, "label":1}`) instead of list (`["image", "label"]`) is because that user can easily resue item (e.g., using `{"image_a":0, "image_b":0, "label":1}`) or skip item (e.g., using `{"image_a":0, "label":2}`).

### How to create custom data reader creator

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

An example implementation of paddle.train could be:

```python
def train(batch_reader, mapping, batch_size, total_pass):
    for pass_idx in range(total_pass):
        for mini_batch in batch_reader(): # this loop will never end in online learning.
            do_forward_backward(mini_batch, mapping)
```
