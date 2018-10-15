# Design Doc: Fluid Data Pipeline

This document is about how Fluid training and inference programs read data.

## Standard Data Format

### Case 1: Data from Files

Consider a Fluid training program, `resnet50.py`, needs to read data from disk:

```bash
cat data | python resnet50.py
```

Since the person who collects the data might be different from the person who wrote resnet50.py:

1. Fluid operators used in `resnet50.py` can recognize the file format of `data`, or, we need a standard data format.
1. These operators need to be able to read from the standard input.

### Case 2: Data from Online Generators

Instead of files, data might come online.  For example:

- Data generator for performance benchmarking.
- Data generator for training models like [GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network).
- The online data stream in production systems, e.g., online advertising.

Consider that 

1. data generators could crash and be restarted (by Kubernetes or other cluster management systems), and
1. the network/pipe connection between could the generator and the trainer may break,

we need

1. the data format is fault-tolerable.

### A Choice: RecordIO

The [RecordIO file format](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/recordio/README.md) is a container of records and is fault-tolerable.  It groups record into *chunks*, and each chunk starts with a magic number and includes its MD5 hash, which allows us to check the consistency skip over damaged chunks, which could be created by generators crashed unexpectedly or broken network connections.

## Discussions

### Other Data Formats

We also considered other data formats, e.g., [SSTable](https://www.igvita.com/2012/02/06/sstable-and-log-structured-storage-leveldb/).  Different from that RecordIO is a container of records, SSTable is a container of key-value pairs.  The fact that training and testing data used with machine learning are records but not key-value pairs inspires us to choose ReocrdIO instead of SSTable.

### Data Reading API

An intuitive solution is to provide the following set of Fluid operators:

```python
rio = fluid.open_recordio("/dev/stdin")
records = fluid.read_recordio(rio, num_records)
fluid.close_recordio(rio)
```

However, it seems (at least right now) that we are going to use another data file format for saving model parameters. And, in the future, we might have to introduce more data formats.  So we'd like to have a lower-level API `open_file` and higher-level API like `create_recordio_reader`, and the program should look like:

```python
rio = fluid.create_recordio_reader(fluid.open_file("/dev/stdin"))
records fluid.read_recordio(rio, num_records)
fluid.close_reader(rio) # This closes the file as well.
```

### Record Format

Usually, each instance contains multiple fields.  For example, each data instance of ResNet includes an image and one or more text labels. In another case of text classification, each instance contains text and its labels.

Data reading operators of Fluid must understand not only the file format, but also the record format, so could it map fields into Fluid variables of various types.  For this purpose, we propose to standardize the record format as a protobuf message. 

```protobuf
message Instance {
  ...
  Field fields = 1;
}
```

Here comes the problem -- how do we define `Field`:

#### Choice 1. 

```protobuf
message Instance {
  enum Type {
    PNG_IMAGE = 0;
    JPG_IMAGE = 1;
    TEXT = 2;
    ...
  }
  message Field {
    required Type type = 1;
    optional bytes png_image = 2;
    optional bytes jpg_image = 3;
    optional bytes text = 4;
  }
  ...
```

However, this would lead to an awkward situation that the raw data types would grow quickly and infinitely. It would also force us to create many operators to parse these many kinds of raw data.  

So we prefer choice 2.

#### Choice 2.

We could reuse [`VarDesc.Type`](https://github.com/PaddlePaddle/Paddle/blob/72ee737f3f3e539e1f4b2a5e819e0d62f362c7b0/paddle/fluid/framework/framework.proto#L95) instead of reinventing `Instance.Type`?

```protobuf
message Instance {
  message Field {
    required VarDesc var_desc = 1;
    required bytes var_value = 2;
  }
}
```

### Data Augmentation

A typical kind of data augmentation is to duplicate each training instance by adding various types of noise, so to train a noise-tolerable model.

It is far from trivial to implement the many augmentation operations as Fluid operators, so we'd adopt a more flexible approach -- write the data augmentation program in arbitrary languages and pipe them up.  For example:

```bash
cat data | go run add_noise.go | python resnet50.py
```

As we use standard data format, `add_noise.go` must be able to decode `data` and encode its outputs into the RecordIO format.  We could provide the Go binding of the RecordIO API.

For quick-n-dirty experiments, we might want to write data augmentation programs as Bash/Awk scripts.  In such case, we want the Bash script to read decoded records from stdin and writes to stdout, without being able to encode/decode RecordIO format.  We can do this by 

1. providing programs to encode/decode records to/from RecordIO files, and
1. base64-encoding records so could we use `\n` as the record separator.

The usage would be

```bash
cat data | recordio_decode | awk -f change_label.awk | recordio_encode | python resnet50.py
```

Please be aware that base64 would increase the data size by up to 30%, but for quick-n-dirty experiments, this should be acceptable. For high-performance/production usages, please feel free to write the data augmentation programs in a production language like C++ and Go, and don't use base64.

### Multiplexing

GPUs usually work much faster than data loading, especially when data come from slow data generators.  To fully utilize GPUs, we might multiplex data generators, which requires a new operator, `fluid.create_multiplex_reader`.

Let us consider the following usage:

1. Run N data generator instances, each writes to a [FIFO](http://man7.org/linux/man-pages/man3/mkfifo.3.html):

   ```bash
   mkfifo ch1 ch2 ... chN
   ./data_generator > ch1 &
   ./data_generator > ch2 &
   ...
   ./data_generator > chN
   ```
   
1. Then we start the Fluid program and let it read from the multiplex of ch1, ..., chN:

   ```bash
   python multiplex.py ch1 ch2 ... chN
   ```
   
The `multiplex.py` file in the above example calls `fluid.create_multiplex_reader`:

```python
rio = fluid.create_multiplex_reader(
          fluid.create_recordio_reader(fluid.open_file("ch1")),
          ...
          fluid.create_recordio_reader(fluid.open_file("chN")))
...
fluid.close_reader(rio)
```
