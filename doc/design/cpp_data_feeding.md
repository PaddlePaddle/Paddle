# C++ Data Feeding

In training with Paddle V2 API, data feeding wholly dependents on Python code. To get rid of the Python environment and achieve the goal of "wrapping the whole training by a while loop op" in Paddle Fluid, a C++ data feeding mechanism is required. 

In this document we show the fundamental design of C++ data feeding process, which includes the data reading, shuffling and batching.

## Reader

A new concept named 'Reader' is introduced. `Reader` is a series of inherited classes which can be hold by our `Variable` and they are used to read or process file data.


### `ReaderBase`

`ReaderBase` is the abstract base class of all readers. It defines the all readers' interfaces.

```cpp
class ReaderBase {
 public:
  explicit ReaderBase(const std::vector<DDim>& shapes) : shapes_(shapes) {
    PADDLE_ENFORCE(!shapes_.empty());
  }
  // Read the next batch of data. (A 'batch' can be only one instance)
  virtual void ReadNext(std::vector<LoDTensor>* out) = 0;
  // Show whether the next bacth exists.
  virtual bool HasNext() const = 0;
  
  // Reinitialize the reader and read the file from the begin.
  virtual void ReInit() = 0;
  
  // Get a certain read in data's shape.
  DDim shape(size_t idx) const;
  // Get shapes of all read in data.
  std::vector<DDim> shapes() const { return shapes_; }
  // Set shapes of read in data.
  void set_shapes(const std::vector<DDim>& shapes) { shapes_ = shapes; }

  virtual ~ReaderBase() {}

 protected:
  std::vector<DDim> shapes_;
};
```

### `FileReader` and `DecoratedReader`

These two classes are derived from the `ReaderBase` and will further be derived by respective specific readers. That is to say, in our design, there are two kinds of readers: file readers and decorated readers. A file reader reads from a file of some specific format, and yield only one instance of data at a time. e.g. RecordIO reader, jpg reader, .... A decorated reader takes another reader(both file reader and decorated reader are OK) as its 'underlying reader'. It gets data from its underlying reader, does some process on them(shuffling, or batching), then yields processed data. The output data of a decorated reader can be a single instance or a batch. `ShuffleReader` and `BatchReader` are both decorated readers.

All the readers share exactly the same interfaces defined in `ReaderBase`. So they can be decorated for more than one time: We can **shuffle** a reader's outputs and then **batch** the shuffle outputs. The interface consistency also allows related ops use readers without knowing what they are exactly.


### `ReaderHolder`

Different readers belong to different class types. It leads to a problem: How can we drop them into `Variable`s and fetch them out by a unified method? For example, if a Variable holds a `BatchReader`, we can not get it by the following code:

```cpp
var->Get<ReaderBase>("batch_reader");
```

we have to write:

```cpp
var->Get<BatchReader>("batch_reader");
```

This requires each time getting a reader from a variable we must know the reader's type exactly. It is nearly impossible.

To solve this problem, we introduce `ReaderHolder` as a wrapper. It acts as an empty decorator of `ReaderBase`, which erases reader's type. With `ReaderHolder` we are able to fetch all types of readers by `var->Get<ReaderHolder>("...")` and regard the obtained object as a reader.

## Related Operators

To create and invoke readers, some now ops are introduced:

### `CreateReaderOp`

Each reader has its creating op. File readers' creating ops have no input and yield the created file reader as its output. Decorated readers' creating ops take the underlying readers as inputs and then yield new decorated readers.

### `ReadOp`

A reader is only a Variable. It cannot trigger the reading process by itself. So we add the `ReadOp` to execute it. A `ReadOp` takes a reader Variable as its input. Each time it runs, it invokes the reader‘s `ReadNext()` function and gets a new batch of data(or only one instance of data, if we use file reader directly). The output data of a reader are in the form of `std::vector<LoDTenosr>`, so the `ReadOp` also needs to split the vector and move LoDTensors to their respective output Variables.
