# Design Doc: Fluid Data Pipeline

This document is about how Fluid training and inference programs reads data.

## Standard Data Format

### Case 1: Data from Files

Consider a Fluid trianing program, `resnet50.py`, needs to read data from disk:

```bash
cat data | python resnet50.py
```

The fact that the who collect the data might not be the same person who wrote `resnet50.py` inspires that

1. Fluid operators used in `resnet50.py` need to recognize the file format of `data`, or, we need a standard data format.
1. These operators need to be able to read the standard input.

### Case 2: Data from Online Generators

Instead of files, data might come online.  For example:

- Data generator for performance benchmarking.
- Data generator for training speical models like [GAN](https://en.wikipedia.org/wiki/Generative_adversarial_network).
- Online data stream in production systems, e.g., online advertising.

Consider that 

1. data generators could crash and be restarted (by Kubernetes or other cluster management systems), and
1. the network/pipe connection between could the generator and the trainer may break,

we need

1. the data format is fault-tolerable.

### A Choice: RecordIO

The [RecordIO file format](https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/fluid/recordio/README.md) is a container of records, and is fault-tolerable.  It groups records into *chunks* and each chunk is attached with the MD5 hash.  So we could check the consistency of a chunk. Also, each chunk starts with a magic number, so we could skip over damaged chunks, which could be created by generators crashed unexpectedly or broken network connections.

## Discussions

### Other Data Formats

We also considered other data formats, e.g., [SSTable](https://www.igvita.com/2012/02/06/sstable-and-log-structured-storage-leveldb/).  Different from that RecordIO is a container of records, SSTable is a container of key-value pairs.  The fact that training and testing data used with machine learning are records but not key-value pairs, inspires us to choose ReocrdIO instead of SSTable.

### Record Format

Usually, each instance contains multiple fields.  For example, each instance of ResNet includes an image and one or more text labels. In another example of text classification, each instance contains a text and its labels.

Data reading operators of Fluid must understand not only the file format, but also the record format, so could it map fields into Fluid variables of various types.  For this purpose, we propose to standardize the record format as a protobuf message:

```protobuf
message Instance {
  enum Type {
    IMAGE = 0;
    ...
  }
  message Field {
    required Type type = 1;
    repeated bytes image = 2; // PNG image
  }
  Field fields = 1;
}
```

***Open Discussion:*** Should we reuse [`VarDesc.Type`](https://github.com/PaddlePaddle/Paddle/blob/72ee737f3f3e539e1f4b2a5e819e0d62f362c7b0/paddle/fluid/framework/framework.proto#L95) instead of reinventing `Instance.Type`?

### Data Augmentation

A typical kind of data augmentation is to duplicate each training instance by adding various kinds of noise, so to train a noise-tolerable model.

It is far from trivial to implement the enomerous number of augmentation operations as Fluid operators, so we'd adopt a more flexible approach -- write the data augmentation program in arbitrary languages and pipe them up.  For example:

```bash
cat data | go run add_noise.go | python resnet50.py
```

Due to that we are using standard data format, `add_noise.go` must be able to decode `data` and encode its outputs into the RecordIO format.  We could provide the Go binding of the RecordIO API.

We might want to further ease the development of data augmentors, say, as Bash/Awk scripts.  In such case, we want the Bash script to read decoded records from stdin and writes to stdout, without being able to encode/decode RecordIO format.  We can do this by 

1. providing programs to encode/decode records to/from RecordIO files, and
1. base64-encode records so to separate records by the special character `\0`.

The usage would be

```bash
cat data | recordio_decode | awk -f change_label.awk | recordio_encode | python resnet50.py
```

Please be aware that base64 would increase the data size by up to 30%, but for quick-n-dirty experiments, this should be acceptable. For high-performance/production usages, please feel free to write the data augmentor in a production language like C++ and Go, and don't use base64.
