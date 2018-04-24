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

### Metadata Storage

### Data Augmentation

