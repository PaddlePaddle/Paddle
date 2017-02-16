# Python Data Provider Design Doc

Data provider provides data for training. It will be passed into `paddle.train` as a parameter.

## Data Provider Interface

Data provider is a function with no parameter that creates a iterable (anything can be used in `for x in iterable`):

```
iterable = data_provider()
```

Element produced for the iterable should be a **single** entry of data, in format `[column_0_item, column_1_item, ...]`. Each element of the list needs to be supported data type (e.g., numpy 1d array of float32, list of int).

For example, `column_0_item` could be image pixels of format numpy 1d array of float32, and `column_1_item` could be image label of format single int value:

```
for single_entry in iterable:
	pixel = entry[0]
	label = entry[1]
```

## Usage

data provider, mapping from data provider column to data layer, batch size and number of total pass will be passed into `paddle.train`:

```python
# two data layer is created:
image_layer = paddle.layer.data("image", ...)
label_layer = paddle.layer.data("label", ...)

# ...

paddle.train(paddle.data.mnist, ["image", "label"], 128, 10, ...)
```
## Q & A

### Why return only a single entry, but not a mini batch?

If return a mini batch, data provider need to take care of batch size. But batch size is a concept for training, it makes more sense for user to specify batch size as a parameter for `train`.

Concretely, always return a single entry make reusing existing data providers much easier (e.g., if existing data provider return not a single entry but 3 entries, training code will be more complex because it need to handle cases like batch size 2).

### How to create custom data provider

```python
def image_provider(path):
	f = open(path)
	images = numpy.fromfile(
		f, 'ubyte', count=n * 28 * 28).reshape((n, 28 * 28)).astype('float32')
	images = images / 255.0 * 2.0 - 1.0
	labels = numpy.fromfile(l, 'ubyte', count=n).astype("int")
	for i in xrange(n):
		yield [images[i, :], labels[i]] # a single entry of data is created each time
	f.close()

# use python lambda to change image_provier into a function with no parameters.
provider = lambda : image_provier("/path/to/data/")
paddle.train(provider, ["image", "label"], ...)
```

### How is `paddle.train` implemented

An example implementation of paddle.train could be:

```python
def make_minibatch_generator(reader, minibatch_size):
    buf = [reader.next() for x in xrange(minibatch_size)]
    while len(buf) > 0:
        yield buf
        buf = [reader.next() for x in xrange(minibatch_size)]

def train(provider, mapping, batch_size, total_pass):
	for pass_idx in range(total_pass):
		for mini_batch in make_minibatch_generator(provider(pass_idx)): # this loop will never end in online learning.
			do_forward_backward(mini_batch, mapping)
```

This is just an example implementation, more complicated logic like data prefetch could be implemented.
