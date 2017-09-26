# Paddle Low Level Data Reader

This design doc will describe a way to implement C++ written data reader(feeder) to
feed data to paddle when train or inference inorder to improve performance, and make
best usage of CPU and GPU usage.

This design can be used for both V2 API and new operator design.

## Background

We have sevaral issues talking about training with v2 API get bad training performance,
like:
- https://github.com/PaddlePaddle/Paddle/issues/3675 and
- https://github.com/PaddlePaddle/Paddle/issues/4156 and
- https://github.com/PaddlePaddle/Paddle/issues/3600.

The current implementations for data reader can greatly simplify user defining how
to parse and feed the data to neural network, but we need to copy data from python
to C++ when feeding the data, and the data feeder will wait until the calculations
of one mini-batch is done.

In V1 implementations there are some considerations to improve data feeding
performance, we can port these features to improve the reader.

## Implementaion

A low-level reader contains below components:

- Data parser:
    A data parser will use `dlopen` to open user-defined parser plugin, and call
    `parse` to convert a line of string to internal `Matrix` or `Tensor` object.
- User parser plugin:
    User must write a C++ plugin to parse data to `Matrix` or `Tensor` object. The
    plugin should implement one interface:
    ```c++
    // return Matrix* for v2 API call
    Tensor* parse(const std::string& line);
    ```
- Buffer:
    A `DoubleBuffer` which is able to async load data when caculations are running.
    We can select to use a general "memory buffer" or "mmap" buffer.
- Thread pool:
    A pool of threads to do async data loading, each thread will call data parser to
    parse and buffer the data, thread will wait if buffer is full. When doing multi-GPU
    training, there should be as much thread as GPU cards.
- Reader settings:
    Low-level reader can be created by calling python API, the settings are passed by
    parameters when constructing the reader:
    ```python
        reader = api.low_level_reader(parser="myparser.so", buffer_size=8192, buffer_type="mmap")
    ```

Then implement a python "trainer" interface to use this low-level reader.

Reference to some legacy implementations at `paddle/gserver/dataproviders/DataProvider.h`.


## Usage

1. Write you own data parser in C++:
    ```c++
    #include "datafeeder.h"

    class MyFeeder : public DataFeeder {
        Tensor* parse(const std::string& line) {
            // your implementaion here
        }
    }
    ```
1. Compile your parser together with paddle header files, generating a library `so` file.
1. Create a reader and use it in your python code like above.
