# Python Data Reader Design Doc

Paddle reads data from data reader during training. It will be passed into `paddle.train` as a parameter.

## Data Reader Interface

Data reader is a function with no parameter that creates a iterable (anything can be used in `for x in iterable`):

```
iterable = data_reader()
```

Element produced for the iterable should be a **single** entry of data, **not** a mini batch. That entry of data could be a single item, or a tuple of items. Item should be of [supported type](http://www.paddlepaddle.org/doc/ui/data_provider/pydataprovider2.html?highlight=dense_vector#input-types) (e.g., numpy 1d array of float32, int, list of int)

An example implementation for single item data reader:

```python
def data_reader_fake_image():
	while True:
		yield numpy.random.uniform(-1, 1, size=20*20)
```

An example implementation for multiple item data reader:
```python
def data_reader_fake_image_and_label():
	while True:
		yield numpy.random.uniform(-1, 1, size=20*20), False
```

## Usage

data reader, mapping from item(s) read to data layer, batch size and number of total pass will be passed into `paddle.train`:

```python
# two data layer is created:
image_layer = paddle.layer.data("image", ...)
label_layer = paddle.layer.data("label", ...)

# ...

paddle.train(paddle.dataset.mnist, {"image":0, "label":1}, 128, 10, ...)
```

## Data Reader Decorators

Data reader decorators takes a single or multiple data reader, returns a new data reader. It is similar to a [python decorator](https://wiki.python.org/moin/PythonDecorators), but it does not use `@` syntax.

Since we have a strict interface for data readers (no parameter, return a single data item). Data reader can be used flexiable via data reader decorators. Following are a few examples:

### Prefetch Data

Since reading data may take time and training can not proceed without data. It is generally a good idea to prefetch data.

Use `paddle.reader.buffered` to prefetch data:

```python
buffered_reader = paddle.reader.buffered(paddle.dataset.mnist, 100)
```

`buffered_reader` will try to buffer (prefetch) `100` data entries.

### Compose Multiple Data Readers

For example, we want to use a source of real images (reusing mnist dataset), and a source of fake images as input for [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661).

We can do:

```python
def data_reader_fake_image():
	while True:
		yield numpy.random.uniform(-1, 1, size=20*20)

def data_reader_bool(t):
	while True:
		yield t

true_reader = lambda : data_reader_bool(True)
false_reader = lambda : data_reader_bool(False)

reader = paddle.reader.combine(paddle.dataset.mnist, data_reader_fake_image, true_reader, false_reader)
# Skipped 1 because paddle.dataset.mnist produces two items per data entry.
# And we don't care second item at this time.
paddle.train(reader, {"true_image":0, "fake_image": 2, "true_label": 3, "false_label": 4}, ...)
```

### Shuffle

Given shuffle buffer size `n`, `paddle.reader.shuffle` will return a data reader that buffers `n` data entries and shuffle them before a data entry is read.

Example:
```python
reader = paddle.reader.shuffle(paddle.dataset.mnist, 512)
```

## Q & A

### Why return only a single entry, but not a mini batch?

If a mini batch is returned, data reader need to take care of batch size. But batch size is a concept for training, it makes more sense for user to specify batch size as a parameter for `train`.

Practically, always return a single entry make reusing existing data reader much easier (e.g., if existing data reader return not a single entry but 3 entries, training code will be more complex because it need to handle cases like batch size 2).

### Why use a dictionary but not a list to provide mapping?

We decided to use dictionary (`{"image":0, "label":1}`) instead of list (`["image", "label"]`) is because that user can easily resue item (e.g., using `{"image_a":0, "image_b":0, "label":1}`) or skip item (e.g., using `{"image_a":0, "label":2}`).

### How to create custom data reader

```python
def image_reader(image_path, label_path):
	f = open(image_path)
	l = open(label_path)
	images = numpy.fromfile(
		f, 'ubyte', count=n * 28 * 28).reshape((n, 28 * 28)).astype('float32')
	images = images / 255.0 * 2.0 - 1.0
	labels = numpy.fromfile(l, 'ubyte', count=n).astype("int")
	for i in xrange(n):
		yield images[i, :], labels[i] # a single entry of data is created each time
	f.close()

# use python lambda to change image_reader into a function with no parameters.
reader = lambda : image_reader("/path/to/image_file", "/path/to/label_file")
paddle.train(reader, {"image":0, "label":1}, ...)
```

### How is `paddle.train` implemented

An example implementation of paddle.train could be:

```python
def minibatch_decorater(reader, minibatch_size):
	def ret():
		r = reader()
		buf = [r.next() for x in xrange(minibatch_size)]
		while len(buf) > 0:
			yield buf
			buf = [r.next() for x in xrange(minibatch_size)]
	return ret

def train(reader, mapping, batch_size, total_pass):
	for pass_idx in range(total_pass):
		for mini_batch in minibatch_decorater(reader): # this loop will never end in online learning.
			do_forward_backward(mini_batch, mapping)
```
