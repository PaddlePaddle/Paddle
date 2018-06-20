# Python Data Feeding

In the former implementation of Paddle Fluid, there are two ways to feed data:

- Use `reader_op` in backend C++ side. This method only supports data feeding from recordio files and random data generators, but supports many kinds of `decorated_readers`. For examples, `double_buffer_reader` uses two threads to achieve better performance: one for time-consuming I/O operations, and the other for `Executor.Run()`. See [C++ Data Feeding](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/cpp_data_feeding.md) for details.

- Feed data directly using `DataFeeder.feed()` in Python codes. It is more flexible than the first way. Many kinds of preprocessing steps can be performed before feeding using Python or any other languages, instead of adding many uncommon `operators` in C++ side. But this method is less efficient: the program cannot read the next mini-batch data before `Executor.Run()` ends. Moreover, `decorated_readers` such as `double_buffer_reader` cannot be used for better performance.

In this document, we design a Python Data Feeding process combining the efficiency of the first way and the flexibility of the second way. A data queue `PyArrayFeedQueue` is designed to be shared by the Python and C++ side, while Python array is pushed into the queue and `reader_op` in C++ side reads out the data from the queue.

## Design of PyArrayFeedQueue
`PyArrayFeedQueue` is a blocking queue with a fixed `capacity` and accepts Python array with shapes indicated by `dims`.
```C++
class PyArrayFeedQueueHolder;

class PyArrayFeedQueue {
  friend class PyArrayFeedQueueHolder;
 private:
  PyArrayFeedQueue(size_t capacity, const std::vector<framework::DDim>& dims, const Place& place);
 public:
  size_t size() const; // Get the current size of the queue
  size_t capacity() const; // Get the capacity of the queue
  bool is_full() const;
  bool is_empty() const;
  
  // Convert Python array tuple to std::vector<framework::LoDTensor> and store it.
  // Block if is_full() == true
  // Use pybind11::gil_scoped_release to release GIL of Python
  void push(const pybind11::tuple& array_tuple);
  
  // Block if is_empty() == true
  // Use pybind11::gil_scoped_release to release GIL of Python
  std::vector<framework::LoDTensor> pop();
 private:
  BlockingQueue<std::vector<framework::LoDTensor>> queue_;
};

class PyArrayFeedQueueHolder {
 public:
  PyArrayFeedQueueHolder() {}
  
  // Calls the constructor of PyArrayFeedQueue to create feeder_
  // For each instance of PyArrayFeedQueueHolder, this function can only called once
  void init_once(size_t capacity, const std::vector<framework::DDim>& dims, const Place& place);
  
  PyArrayFeedQueue& feeder(); // Get feeder_
  const PyArrayFeederQueue& feeder() const; // Get feeder_
 private:
  std::unique_ptr<PyArrayFeedQueue> feeder_;
};
```

There are some major things that must be concerned:
- `PyArrayFeedQueueHolder` should be a `Variable` in global scope, so that `reader_op` can find it when reading data. Since `PyArrayFeedQueue` does not have a default constructor, it cannot be constructed by `Scope::Var()::GetMutable<T>()`. To solve this problem, `PyArrayFeedQueueHolder` is designed to defer construction of `PyArrayFeedQueue`.
- A `Variable` of `PyArrayFeedQueueHolder` but not `VarDesc` must be created in Python code before `Executor.Run()` so that `Executor.Run()` can get the feeding data when it is called.
- `Create_reader_op` should accept the name or address of `PyArrayFeedQueueHolder` as an input or attribute.


## Design of PyArrayReader
`PyArrayReader` is a reader which holds a `PyArrayFeedQueue` object. Notice that `ReInit()` function is not supported because the capacity of the `PyArrayFeedQueue` object is limited.
```C++
class PyArrayReader : public ReaderBase {
 public:
  explicit PyArrayReader(PyArrayFeedQueue* queue);
  
  void ReadNext(std::vector<framework::LoDTensor>* out) override;
  
  void ReInit() override {
    PADDLE_THROW("PyArrayReader does not support ReInit()");
  }
 private:
  PyArrayFeedQueue* queue_;
};
```

## Design of CreatePyArrayReaderOp
`CreatePyArrayReaderOp` is used to create `PyArrayReader` object. It requires an attribute of `feeder_name` which indicates the name of the `PyArrayFeedQueueHolder` variable.
```C++
class CreatePyArrayReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;
 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    const std::string& feeder_name = Attr<std::string>("feeder_name");
    auto* feeder_holder_var = scope.FindVar(feeder_name);
    PADDLE_ENFORCE(feed_holder_var != nullptr);
    auto* feeder_holder = feeder_holder_var
                    ->template GetMutable<framework::PyArrayFeedQueueHolder>();
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    out->Reset(new PyArrayReader(feeder_holder->feeder());
  }
};
```

## Design of Python codes
The design of Python codes are as follows. First, we construct a variable of `PyArrayFeedQueueHolder` and init it with given parameters, returning the `PyArrayFeedQueue` object after initialization. After that, a layer of `CreatePyArrayReaderOp` is constructed and accepts the name of the `PyArrayFeedQueueHolder` variable. The `PyArrayFeedQueue` object and result of the layer are both returned.
```Python
def py_array_reader(capacity, shapes, place):
  feeder_name = unique_name.generate("py_array_feed_queue")
  var = global_scope().var(feeder_name) # create PyArrayFeedQueueHolder Variable
  feed_queue = core.init_py_array_feed_queue(var, capacity, shapes, place) # init PyArrayFeedQueue
  out = create_var()
  create_reader_op_with_feeder_name(
      type='create_py_array_reader',
      outputs={'Out':[out]},
      attrs = {'feeder_name': feeder_name})  
  return out, feed_queue
```
