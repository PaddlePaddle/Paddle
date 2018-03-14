# C++ Data Feeding

While using Paddle V2 API for training, data feeding completely depends on the Python code. To get rid of the Python environment and achieve the goal of "wrapping the whole training by a while loop op" in Paddle Fluid, a C++ data feeding mechanism is required.

In this document we show the fundamental design of a C++ data feeding process, which includes data reading, shuffling and batching.

## Reader

In order to handle the above mentioned problem, a new concept called 'Reader' is introduced. `Reader` is a series of inherited classes which can be held by our `Variable` and they are used to read or process file data.


### `ReaderBase`

`ReaderBase` is the abstract base class for all readers. It defines the interface for all readers.

```cpp
class ReaderBase {
 public:
  // Reads the next batch of data. (A 'batch' can be only one instance)
  // If the next batch doesn't exist, it throws an exception
  virtual void ReadNext(std::vector<LoDTensor>* out) = 0;
  
  // Checks whether the next instance exists.
  virtual bool HasNext() = 0;
  
  // Reinitializes the reader and read the file from the beginning.
  virtual void ReInit() = 0;

  virtual ~ReaderBase() {}
};
```

### FileReader

`FileReader` is derived from the `ReaderBase`. It is still an abstract class and will further be derived by Readers of respective specific format.

```cpp
class FileReader : public ReaderBase {
 public:
  explicit FileReader(const std::vector<DDim>& shapes) : shapes_(shapes) {}

  void ReadNext(std::vector<LoDTensor>* out) override final {
    ReadNextImpl(out);
    CheckShapes(out);
  }

  virtual void ReadNextImpl(std::vector<LoDTensor>* out) = 0;

 protected:
  // Checks whether the out shapes is consistent with shapes_
  CheckShape(const std::vector<LoDTensor>* out);

  std::vector<DDim> shapes_;
};
```

A file reader binds with a single file, and reads one instance of data from the file at a time. Each type of file reader shall implement its own `ReadNextImpl()`, `HasNext()` and `ReInit()`.

### DecoratedReader

A decorated reader takes another reader(both file reader and decorated reader are OK) as its 'underlying reader'. It gets data from its underlying reader, does some process on them(shuffling,  batching or something else), then yields processed data. The output data of a decorated reader can be a single instance or a batch. `ShuffleReader` and `BatchReader` are both decorated readers.

```cpp
class DecoratedReader : public ReaderBase {
 public:
  explicit DecoratedReader(ReaderBase* reader) : reader_(reader) {
    PADDLE_ENFORCE_NOT_NULL(reader_);
  }

  void ReInit() override { reader_->ReInit(); }

 protected:
  ReaderBase* reader_;
};
```

All the `FileReader` and `DecoratedReader` share exactly the same interfaces as defined in `ReaderBase`. So they can be decorated for more than one time: We can **shuffle** a reader's outputs and then **batch** the shuffle outputs. The interface consistency also allows related ops use readers without knowing what they are exactly.

### ThreadedReader


### `ReaderHolder`

Different readers belong to different class types. This leads to a problem: How can we drop them into `Variable`s and fetch them out by a unified method? For example, if a Variable holds a `BatchReader`, we can not get it by the following code:

```cpp
var->Get<ReaderBase>("batch_reader");
```

We would have to write:

```cpp
var->Get<BatchReader>("batch_reader");
```

This requires that in order to get a reader from a variable, every time, we must know the reader's type exactly. This is nearly impossible.

To solve this problem, we introduce `ReaderHolder` as a wrapper. It acts as an empty decorator of `ReaderBase`, which hides reader's type. With `ReaderHolder` we are able to fetch all types of readers by `var->Get<ReaderHolder>("...")` and regard the obtained object as a reader.

## Related Operators

To create and invoke readers, some new ops are introduced:

### `CreateReaderOp`

Each reader has its creation op. File readers' creation ops have no input and yield the created file reader as its output. Decorated readers' creation ops take the underlying readers as inputs and then yield new decorated readers.

### `ReadOp`

A reader is only a Variable. It cannot trigger the reading process by itself. So we add the `ReadOp` to execute it. A `ReadOp` takes a reader Variable as its input. Each time it runs, it invokes the reader‘s `ReadNext()` function and gets a new batch of data(or only one instance of data, if we use file reader directly). The output data of a reader are in the form of `std::vector<LoDTenosr>`, so the `ReadOp` also needs to split the vector and move LoDTensors to their respective output Variables.
