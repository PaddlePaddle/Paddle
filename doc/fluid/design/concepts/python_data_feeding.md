# Python Data Feeding

In the former implementation of Paddle Fluid, there are two ways to feed data:

- Use `reader_op` in backend C++ side. This method only supports data feeding from recordio files and random data generators, but supports many kinds of `decorated_readers`. For examples, `double_buffer_reader` uses two threads to achieve better performance: one for time-consuming I/O operations, and the other for `Executor::Run()`. See [C++ Data Feeding](https://github.com/PaddlePaddle/Paddle/blob/develop/doc/fluid/design/concepts/cpp_data_feeding.md) for details.

- Feed data directly using `DataFeeder.feed()` in Python codes. It is more flexible than the first way. Many kinds of preprocessing steps can be performed before feeding using Python or any other languages, instead of adding many uncommon `operators` in C++ side. But this method is less efficient: the program cannot read the next mini-batch data before `Executor::Run()` ends. Moreover, `decorated_readers` such as `double_buffer_reader` cannot be used for better performance.

In this document, we design a Python Data Feeding process combining the efficiency of the first way and the flexibility of the second way. A data queue `LoDTensorBlockingQueue` is designed to be shared by the Python and C++ side, while `LoDTensorArray` is pushed into the queue in Python side and `reader_op` in C++ side reads out the data from the queue.


## Design of LoDTensorBlockingQueue
`LoDTensorBlockingQueue` is a blocking queue with a fixed `capacity` and accepts `std::vector<framework::LoDTensor>` with shapes indicated by `dims`. Since `LoDTensorBlockingQueue` must be constructed using `capacity` and `dims`, it cannot be a `Variable` type. Therefore, a `LoDTensorBlockingQueueHolder` is designed to defer construction of `LoDTensorBlockingQueue`.

```C++
class LoDTensorBlockingQueueHolder;

class LoDTensorBlockingQueue {
  friend class LoDTensorBlockingQueueHolder;
 private:
  // `LoDTensorBlockingQueue` can only be constructed by 
  // `LoDTensorBlockingQueueHolder::InitOnce()`
  LoDTensorBlockingQueue(size_t capacity, const std::vector<framework::DDim>& dims);
 
 public:
  size_t Size() const { return queue_.Size(); } // Get the current size of the queue

  size_t Cap() const { return queue_.Cap(); }// Get the capacity of the queue

  void Close() { return queue_.Close(); }

  bool IsClosed() const { return queue_.IsClosed(); }

  // Block if Size() == Cap()
  // Return false only when queue_.IsClosed() == true
  bool Push(const std::vector<framework::LoDTensor> &lod_tensor_vec);
  
  // Block if Size() == 0.
  // *Success == false when queue_.IsClosed() == true
  std::vector<framework::LoDTensor> Pop(bool *success = nullptr);
 
 private:
  // Use reader::BlockingQueue as the inner data structure
  BlockingQueue<std::vector<framework::LoDTensor>> queue_;
  std::vector<framework::DDim> dims_;
};

class LoDTensorBlockingQueueHolder {
 public:  
  // Call the constructor of `LoDTensorBlockingQueue` to create queue_
  // `InitOnce` can only called once, otherwise an exception would raise
  void InitOnce(size_t capacity, const std::vector<framework::DDim>& dims) {
    PADDLE_ENFORCE(queue_ == nullptr);
    queue_.reset(new LoDTensorBlockingQueue(capacity, dims));
  }

  const std::shared_ptr<LoDTensorBlockingQueue>& GetQueue() const { return queue_; }

 private:
  std::shared_ptr<LoDTensorBlockingQueue> queue_;
};
```

There are some major things that must be concerned:
- `LoDTensorBlockingQueueHolder` should be a `Variable` in global scope, so that `reader_op` can find it when reading data.
- A `Variable` of `LoDTensorBlockingQueueHolder` but not `VarDesc` must be created in Python code before `Executor::Run()` so that `Executor::Run()` can get the feeding data when it is called.
- `Create_reader_op` should accept the name of the `LoDTensorBlockingQueueHolder` variable as an input.


## Release of the GIL in pybind
`Pybind11::gil_scoped_release` is used to release GIL (Global Interpreter Lock) when `LoDTensorBlockingQueue::Push()` or `Executor::Run()` method are invoked in Python side, making `LoDTensorBlockingQueue::Push()` and `Executor::Run()` run in parallel.


## Design of PyReader
`PyReader` is a reader which holds a `LoDTensorBlockingQueue` object.
```C++
class PyReader : public ReaderBase {
 public:
  explicit PyReader(const std::shared_ptr<LoDTensorBlockingQueue>& queue);
  
  void ReadNext(std::vector<framework::LoDTensor>* out) override {
    bool success;
    *out = queue_->Pop(&success);
    if (!success) out->clear();
  }
  
  void ReInit() override { return; }

 private:
  std::shared_ptr<LoDTensorBlockingQueue> queue_;
};
```


## Design of CreatePyReaderOp
`CreatePyReaderOp` is used to create the `PyReader` object. It requires an input `blocking_queue` which indicates the name of the `LoDTensorBlockingQueueHolder` variable.
```C++
class CreatePyReaderOp : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;
 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    auto* out = scope.FindVar(Output("Out"))
                    ->template GetMutable<framework::ReaderHolder>();
    if (out->Get() != nullptr) return;
    
    const std::string& queue_name = Input("blocking_queue");
    auto* queue_holder_var = scope.FindVar(queue_name);
    PADDLE_ENFORCE(queue_holder_var != nullptr);
		auto* queue_holder = queue_holder_var
                    ->template GetMutable<framework::LoDTensorBlockingQueueHolder>();
    out->Reset(new PyReader(queue_holder->GetQueue()));
  }
};
```

## Design of Python codes
The design of Python codes are as follows. First, we construct a variable of `LoDTensorBlockingQueueHolder` and init it with given parameters, returning the `LoDTensorBlockingQueue` object after initialization. After that, a layer of `CreatePyReaderOp` is constructed and accepts the name of the `LoDTensorBlockingQueueHolder` variable. The `LoDTensorBlockingQueue` object and result of the layer are both returned.
```Python
def py_reader(capacity, shapes):
  queue_name = unique_name.generate("lod_tensor_blocking_queue")
  var = global_scope().var(feeder_name) # create LoDTensorBlockingQueueHolder Variable
  feed_queue = core.init_lod_tensor_blocking_queue(var, capacity, shapes) # init the queue
  out = create_var()
  create_py_reader_op_with_queue_name(
      inputs={'blocking_queue': queue_name},
      outputs={'Out':[out]})  
  return out, feed_queue
```
