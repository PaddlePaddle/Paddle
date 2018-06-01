/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifndef PADDLE_NO_PYTHON

#include <Python.h>
#include <numpy/numpyconfig.h>
#include <stdio.h>
#include <stdlib.h>
#include <list>
#include <unordered_set>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#include "DataProvider.h"

#include "paddle/utils/Locks.h"
#include "paddle/utils/PythonUtil.h"
#include "paddle/utils/Stat.h"

namespace paddle {

namespace unittest {

static std::unique_ptr<std::function<void(size_t /*poolActualSize */)>>
    OnPoolFilled;

namespace pydp2 {

void setOnPoolFilledHook(const std::function<void(size_t)>& callback) {
  OnPoolFilled.reset(new std::function<void(size_t)>());
  *OnPoolFilled = callback;
}

void clearOnPoolFilledHook() { OnPoolFilled.reset(); }

}  // namespace pydp2
}  // namespace unittest

/**
 * Slot type
 */
enum SlotType {
  ST_DENSE = 0,
  ST_NON_SPARSE_VALUE = 1,
  ST_SPARSE_VALUE = 2,
  ST_INDEX = 3
};

/**
 * Sequence type
 */
enum SeqType { SQT_NONE = 0, SQT_SEQ, SQT_SUBSEQ };

/**
 * Cache Type.
 */
enum CacheType {
  NO_CACHE = 0,           // Each pass will load data from PyDataProvider2.
  CACHE_PASS_IN_MEM = 1,  // First pass will load data from PyDataProvider2,
                          // then cache all data in memory. Load data from
                          // memory in rest passes.
};

struct SlotHeader {  // Slot Header will parse from python object's slots field.
  size_t dim;
  SlotType slotType;
  SeqType seqType;
};

inline std::ostream& operator<<(std::ostream& os, const SlotHeader& header) {
  os << "Dim = " << header.dim << " Type = " << header.slotType
     << " SeqType = " << header.seqType;
  return os;
}

/**
 * FieldScanner Interface.
 *
 * It will read python object, and fill to argument's each slot.
 * There are two steps, prepare and fill. Scanner will alloc memory during
 * prepare step, fill data into argument during fill step.
 */
class IFieldScanner {
 public:
  DISABLE_COPY(IFieldScanner);
  /**
   * Ctor.
   * @param headerPtr slot header that scanner belong to.
   */
  explicit IFieldScanner(SlotHeader* headerPtr) : headerPtr_(headerPtr) {}
  virtual ~IFieldScanner() {}

  /**
   * Start prepare step.
   */
  virtual void startPrepare(Argument& argument) {}

  /**
   * Prepare step.
   *
   * @note the obj could be a timestep of sample or whole sample. It depends
   * what scanner it is.
   */
  virtual void prepare(Argument& argument, PyObject* obj) {}

  /**
   * Finish Prepare step.
   */
  virtual void finishPrepare(Argument& argument) {}

  /**
   * Start fill step.
   */
  virtual void startFill(Argument& argument) {}

  /**
   * Fill step.
   *
   * @note the obj could be a timestep of sample or whole sample. It depends
   * what scanner it is.
   */
  virtual void fill(Argument& argument, PyObject* obj) {}

  /**
   * Finish fill step.
   */
  virtual void finishFill(Argument& argument) {}

  /**
   * Factory method. Create a scanner by header. The final scanner may be
   * combine many scanners.
   *
   * @note Fatal if header is not support.
   */
  static IFieldScanner* create(SlotHeader* header);

 protected:
  SlotHeader* headerPtr_;
};

/**
 * Py Data Provider Cache Interface.
 */
class IPyDataProviderCache {
 public:
  virtual ~IPyDataProviderCache() {}

  /**
   * invoke when DataProvider::reset()
   * @return true if read data from python.
   */
  virtual bool reset() = 0;

  /**
   * invoke when these data are used by DataProvider, and need to clear.
   * @param [inout] data used data.
   *
   * @note The implemented class must clear these data array. Or if you want to
   * delete the PyObjectPtr later, you should make sure the paddle process only
   * have one active thread calling python code (use PyGuard otherwise).
   */
  virtual void drop(std::deque<PyObjectPtr>* data) = 0;

  /**
   * Return whole data in cache.
   */
  virtual std::deque<PyObjectPtr>* load() = 0;

  /**
   * Factory method. Convert CacheType to IPyDataProviderCache*
   */
  static IPyDataProviderCache* create(CacheType ct);
};

/**
 * PyDataProvider2.
 *
 * For usage, please refer python module 'paddle.trainer.PyDataProvider2'
 *
 * Here, we start a thread to read data. It is totally asynchronous for reading
 * data. And it support cache strategies.
 */
class PyDataProvider2 : public DataProvider {
 public:
  /**
   * Ctor
   */
  PyDataProvider2(const DataConfig& config,
                  const ModelConfig& modelConfig,
                  bool useGpu)
      : DataProvider(config, useGpu), callingContextCreated_(2) {
    if (PyArray_API == NULL) import_array();
    auto& args = config.load_data_args();
    PyObjectPtr kwargs = PyObjectPtr(PyDict_New());
    if (!args.empty()) {
      kwargs = callPythonFuncRetPyObj(
          "paddle.trainer.PyDataProvider2", "deserialize_args", {args});
    }

    py::DictHelper kwargsDict(kwargs);
    kwargsDict.setBool("is_train", !config.for_test());
    std::vector<std::string> inputs;
    inputs.reserve(modelConfig.input_layer_names().size());
    std::copy(modelConfig.input_layer_names().begin(),
              modelConfig.input_layer_names().end(),
              std::back_inserter(inputs));
    kwargsDict.setStringList("input_order", inputs);

    // kwargs is keyword arguemts to create object.
    this->createPyDataObj(config.load_data_module(),
                          config.load_data_object(),
                          config.files(),
                          std::move(kwargs));
    DBG << "Instance " << instance_.get() << " loaded.";
    this->readPyFields(config.for_test());
    DBG << "Py Field Done";
  }

  /**
   * Dtor
   * @note will stop loading thread when destructing
   */
  virtual ~PyDataProvider2() { resetImpl(false); }

 private:
  void createPyDataObj(const std::string& model,
                       const std::string& className,
                       const std::string& fileListName,
                       PyObjectPtr&& kwargs  // NOLINT
                       ) {
    LOG(INFO) << "loading dataprovider " << model << "::" << className;

    PyObjectPtr module = py::import(model);
    PyObjectPtr moduleDict(PyModule_GetDict(module.get()));
    CHECK_PY(moduleDict) << "Invoke module.__dict__ error";
    PyObjectPtr cls(PyDict_GetItemString(moduleDict.get(), className.c_str()));
    CHECK_PY(cls) << "load class " << className.c_str() << "error";

    // If there are multiple python instance share same module, the PyObjectPtr
    // only for instance will make python reference-count error.
    //
    // So here, we increase reference count manually.
    Py_XINCREF(module.get());
    Py_XINCREF(moduleDict.get());
    Py_XINCREF(cls.get());

    PyObjectPtr fileListInPy = loadPyFileLists(fileListName);
    PyDict_SetItemString(kwargs.get(), "file_list", fileListInPy.get());
    {
      PyGuard guard;
      instance_.reset(PyObject_Call(cls.get(), zeroTuple_.get(), kwargs.get()));
    }
    CHECK_PY(instance_) << "Cannot Create instance";
  }

  void readPyFields(bool testing) {
    py::ObjectHelper self(this->instance_);
    bool ok;

    this->skipShuffle_ =
        !self.getBoolAttr("should_shuffle", &ok /*isBoolType*/);
    if (!ok) {
      this->skipShuffle_ = testing;  // shuffle when is training, skip shuffle
                                     // when is testing.
    }
    DBG << "Provider Skip Shuffle " << this->skipShuffle_;

    this->poolSize_ = self.getIntAttr<size_t>("pool_size", &ok);
    if (!ok) {
      this->poolSize_ = -1UL;
    }
    this->minPoolSize_ = self.getIntAttr<size_t>("min_pool_size", &ok);
    if (!ok) {
      this->minPoolSize_ = -1UL;
    }
    this->minPoolSize_ = std::min(this->poolSize_, this->minPoolSize_);

    this->canOverBatchSize_ = self.getBoolAttr("can_over_batch_size");

    calcBatchSize_.reset(self.getAttr("calc_batch_size"));
    if (this->calcBatchSize_ && !py::isCallable(this->calcBatchSize_)) {
      this->calcBatchSize_.reset();
    }

    generator_.reset(self.getAttr("generator"));
    CHECK(py::isCallable(generator_));

    // Reading slots.
    PyObjectPtr slotsPtr(self.getAttr("slots"));
    py::SequenceHelper slots(slotsPtr);
    headers_.reserve(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
      headers_.emplace_back();
      auto& header = headers_.back();
      PyObject* hdPtr = slots[i];
      CHECK(hdPtr != nullptr);
      Py_XINCREF(hdPtr);
      PyObjectPtr headerPtrWrap(hdPtr);
      py::ObjectHelper hd(headerPtrWrap);
      header.dim = hd.getIntAttrWithError<size_t>("dim");
      header.seqType = (SeqType)hd.getIntAttrWithError<int>("seq_type");
      header.slotType = (SlotType)hd.getIntAttrWithError<int>("type");
    }

    DBG << "Data header size " << headers_.size();
    for (auto& header : headers_) {
      DBG << header;
    }
    cache_.reset(IPyDataProviderCache::create(
        (CacheType)self.getIntAttrWithError<int>("cache")));
  }

  PyObjectPtr loadPyFileLists(const std::string& fileListName) {
    loadFileList(fileListName, fileLists_);
    PyObject* lst = PyList_New(fileLists_.size());
    for (size_t i = 0; i < fileLists_.size(); ++i) {
      PyList_SET_ITEM(lst, i, PyString_FromString(fileLists_[i].c_str()));
    }
    return PyObjectPtr(lst);
  }

  void loadThread() {
    DBG << "Creating context";
    for (auto& filename : fileLists_) {
      PyGuard g;
      py::CallableHelper generator(this->generator_);
      generator.setArgsSize(2);
      generator.getArgs().set(0, instance_);
      generator.getArgs().set(1, PyString_FromString(filename.c_str()), true);
      callingContexts_.emplace_back(generator());
      CHECK_PY(callingContexts_.back()) << "Generator error.";
      CHECK(PyIter_Check(callingContexts_.back()));
    }
    DBG << "Create context done";
    callingContextCreated_.wait();

    PositionRandom p(skipShuffle_);

    while (!exit_ && !callingContexts_.empty()) {
      PyObject* data = nullptr;

      {  // Read data.
        size_t cid = p(callingContexts_.size());
        bool atEnd;
        data = py::iterNext(callingContexts_[cid], &atEnd);
        if (atEnd || data == nullptr) {
          if (cid != 0) {
            std::swap(callingContexts_[cid], callingContexts_[0]);
            cid = 0;
          }

          PyObjectPtr front;
          {
            std::unique_lock<std::mutex> l(mtx_);
            front = pop_get_front(callingContexts_);
          }
          {
            PyGuard g;
            front.reset();
          }
          this->pullCV_.notify_all();
          continue;
        }
      }

      size_t additionalBatchSize = 1;
      if (calcBatchSize_) {
        PyGuard guard;
        py::CallableHelper calcBatchSize(this->calcBatchSize_);
        calcBatchSize.setArgsSize(1);
        calcBatchSize.getArgs().set(0, data);
        PyObjectPtr bs(calcBatchSize());
        CHECK_PY(bs);
        bool ok;
        additionalBatchSize = py::castInt<size_t>(bs.get(), &ok);
        CHECK(ok) << "CalcBatchSize must return int or long";
      }

      if (this->loadThread_) {  // wait poolActualSize < poolSize;
        std::unique_lock<std::mutex> l(mtx_);
        pushCV_.wait(l, [this] { return this->poolActualSize_ < poolSize_; });
      }

      {
        std::lock_guard<std::mutex> guard(mtx_);
        poolActualSize_ += additionalBatchSize;
        dataPool_.emplace_back(data);
      }
      pullCV_.notify_all();
    }
    DBG << "load thread end";
  }

  inline void resetImpl(bool startNewThread) {
    DBG << "Reseting " << startNewThread;
    exit_.store(true);
    if (loadThread_) {  // is loading.
      loadThread_->join();
      loadThread_.reset();
    }
    {
      PyGuard g;
      callingContexts_.clear();
      this->pullCV_.notify_one();
    }

    std::lock_guard<std::mutex> guard(mutexForReset_);
    {
      PyGuard g;
      dataPool_.clear();
    }
    poolActualSize_ = 0;

    if (startNewThread && cache_->reset()) {
      DBG << "Start new thread.";
      loadThread_.reset(new std::thread([this] {
        exit_ = false;
        loadThread();
      }));
      callingContextCreated_.wait();
    }
    DBG << "Reset done";
    exit_ = false;
  }

 private:
  std::unique_ptr<std::thread> loadThread_;
  std::atomic<bool> exit_;
  std::deque<PyObjectPtr> callingContexts_;
  std::deque<PyObjectPtr> dataPool_;
  size_t poolActualSize_;
  std::condition_variable pushCV_;
  std::condition_variable pullCV_;
  std::mutex mtx_;

  std::mutex mutexForReset_;

  ThreadBarrier callingContextCreated_;
  std::unique_ptr<IPyDataProviderCache> cache_;

  PyObjectPtr instance_;
  size_t poolSize_;
  size_t minPoolSize_;
  bool canOverBatchSize_;
  PyObjectPtr calcBatchSize_;
  PyObjectPtr generator_;
  std::vector<std::string> fileLists_;
  std::vector<SlotHeader> headers_;
  static PyObjectPtr zeroTuple_;

  class PositionRandom {
   public:
    inline explicit PositionRandom(bool skipRand)
        : eng_(ThreadLocalRandomEngine::get()), skipRand_(skipRand) {}

    inline size_t operator()(size_t len) {
      if (!skipRand_) {
        if (!dist_ || dist_->b() != len - 1) {
          dist_.reset(new std::uniform_int_distribution<size_t>(0, len - 1));
        }
        return (*dist_)(eng_);
      } else {
        return 0;
      }
    }

   private:
    std::default_random_engine& eng_;
    std::unique_ptr<std::uniform_int_distribution<size_t>> dist_;
    bool skipRand_;
  };

  // DataProvider interface
 public:
  /**
   * Resetting the PyDataProvider. May start reading thread here.
   */
  virtual void reset() {
    resetImpl(true);
    DataProvider::reset();
  }

  /**
   * Shuffle. Do nothing because PyDataProvider do shuffle implicitly by random
   * select data from datapool.
   */
  void shuffle() {}

  /**
   * Not limited size.
   */
  int64_t getSize() { return -1; }

  /**
   * Loading a batch of data.
   */
  int64_t getNextBatchInternal(int64_t size_, DataBatch* batch) {
    std::lock_guard<std::mutex> guard(mutexForReset_);
    REGISTER_TIMER("PyDP2.getNextBatchInternal")
    CHECK_GE(size_, 0);
    size_t size = (size_t)size_;
    if (loadThread_) {  // loading from thread should wait for data pool ready.
                        // but, loading from cache, cache object should ensure
                        // data pool ready.
      std::unique_lock<std::mutex> l(mtx_);
      pullCV_.wait(l, [this, &size] {
        return this->poolActualSize_ >= std::max(size, this->minPoolSize_) ||
               callingContexts_.empty();
      });

      if (unittest::OnPoolFilled) {
        (*unittest::OnPoolFilled)(this->poolActualSize_);
      }
    }
    std::deque<PyObjectPtr> data;
    size_t bsize = 0;
    std::deque<PyObjectPtr>* poolPtr = nullptr;

    if (this->loadThread_) {  // loading from thread.
      poolPtr = &this->dataPool_;
    } else {  // loading from cache.
      poolPtr = this->cache_->load();
    }
    if (exit_) {
      // PyDataProvider is destructing.
      return 0;
    }
    CHECK(poolPtr != nullptr);

    std::deque<PyObjectPtr>& pool = *poolPtr;

    while (bsize < size && !pool.empty()) {
      {
        // move data from pool to data
        std::lock_guard<std::mutex> guard(mtx_);
        if (skipShuffle_) {
          size_t i = 0;
          CHECK(pool[i] != nullptr);
          data.emplace_back(std::move(pool[i]));
          pool.pop_front();
        } else {  // when shuffle, use swap to drop only last pool element.
          size_t i = ThreadLocalRand::rand() % pool.size();
          CHECK(pool[i] != nullptr);
          if (i != 0) {
            std::swap(pool[i], pool.front());
          }
          data.emplace_back(std::move(pool.front()));
          pool.pop_front();
        }

        if (calcBatchSize_) {  // custom calc batch size.
          PyGuard guard;
          Py_INCREF(data.back().get());
          py::CallableHelper calcBatchSize(calcBatchSize_);
          calcBatchSize.setArgsSize(1);
          calcBatchSize.getArgs().set(0, data.back());
          PyObjectPtr customBatchSize(calcBatchSize());
          bool ok;
          size_t tmp = py::castInt<size_t>(customBatchSize.get(), &ok);
          CHECK(ok) << "calc_batch_size must return int";

          if (bsize + tmp > size && !canOverBatchSize_) {
            // Put data back.
            pool.push_front(std::move(data.back()));
            data.pop_back();
            break;
          } else {
            bsize += tmp;
          }
        } else {
          bsize += 1;
        }
      }
    }

    if (this->loadThread_) {
      {
        std::lock_guard<std::mutex> g(mtx_);
        poolActualSize_ -= bsize;
      }
      this->pushCV_.notify_all();
    }

    if (bsize == 0) {  // end of pass. In data pool, cannot get any data.
      return 0;
    }

    DataBatch cpuBatch;
    cpuBatch.setSize(bsize);
    auto& inArgs = cpuBatch.getStreams();
    inArgs.resize(headers_.size());
    std::vector<std::unique_ptr<IFieldScanner>> scanners;
    scanners.reserve(headers_.size());
    for (auto& header : headers_) {
      scanners.emplace_back(IFieldScanner::create(&header));
    }
    DBG << "Scanner created.";
    for (size_t i = 0; i < headers_.size(); ++i) {
      scanners[i]->startPrepare(inArgs[i]);
    }
    for (auto& d : data) {
      py::SequenceHelper s(d);
      for (size_t i = 0; i < headers_.size(); ++i) {
        scanners[i]->prepare(inArgs[i], s[i]);
      }
    }
    for (size_t i = 0; i < headers_.size(); ++i) {
      scanners[i]->finishPrepare(inArgs[i]);
    }
    for (size_t i = 0; i < headers_.size(); ++i) {
      scanners[i]->startFill(inArgs[i]);
    }
    for (auto& d : data) {
      py::SequenceHelper s(d);
      for (size_t i = 0; i < headers_.size(); ++i) {
        scanners[i]->fill(inArgs[i], s[i]);
      }
    }

    for (size_t i = 0; i < headers_.size(); ++i) {
      scanners[i]->finishFill(inArgs[i]);
    }

    {
      PyGuard g;
      cache_->drop(&data);
    }

    DBG << "Reading CPU Batch Done.";

    if (useGpu_) {
      std::vector<Argument>& cpuArguments = cpuBatch.getStreams();
      DataBatch& gpuBatch = *batch;
      std::vector<Argument>& gpuArguments = gpuBatch.getStreams();
      gpuArguments.resize(cpuArguments.size());
      gpuBatch.setSize(bsize);
      for (size_t i = 0; i < headers_.size(); ++i) {
        gpuArguments[i].resizeAndCopyFrom(
            cpuArguments[i], useGpu_, HPPL_STREAM_1);
      }
      hl_stream_synchronize(HPPL_STREAM_1);
    } else {
      *batch = cpuBatch;
    }
    return bsize;
  }
};

PyObjectPtr PyDataProvider2::zeroTuple_(PyTuple_New(0));

REGISTER_DATA_PROVIDER_EX(py2, PyDataProvider2);

/**
 * Scanner for dense slot.
 */
class DenseScanner : public IFieldScanner {
 public:
  explicit DenseScanner(SlotHeader* ptr) : IFieldScanner(ptr), height_(0) {}

  /**
   * Prepare.
   * @param argument target argument
   * @param obj each timestep of a sample.
   */
  virtual void prepare(Argument& argument, PyObject* obj) { ++height_; }

  virtual void finishPrepare(Argument& argument) {
    Matrix::resizeOrCreate(
        argument.value, height_, headerPtr_->dim, false, false);
    height_ = 0;
  }

  /**
   * Fill argument from obj.
   * @param argument
   * @param obj
   */
  virtual void fill(Argument& argument, PyObject* obj) {
    real* dat = argument.value->getData() + height_ * headerPtr_->dim;
    if (PyArray_Check(obj)) {
      auto dtype = PyArray_DTYPE((PyArrayObject*)obj);
      if (dtype->type == 'f' && dtype->elsize == sizeof(real)) {
        real* data = (real*)PyArray_DATA((PyArrayObject*)obj);
        auto sz = PyArray_SIZE((PyArrayObject*)obj);
        std::copy(data, data + sz, dat);
      } else {
        LOG(FATAL) << "You should yield float" << sizeof(real) * 8 << " array";
      }
    } else {
      py::SequenceHelper s(obj);
      // TODO(yuyang18): Here we can use AVX or SSE to accelerate memory copy.
      for (size_t i = 0; i < headerPtr_->dim; ++i) {
        dat[i] = (real)s.getDouble(i);
      }
    }
    ++height_;
  }

 private:
  size_t height_;
};

/**
 * Scanner for index slot
 */
class IndexScanner : public IFieldScanner {
 public:
  explicit IndexScanner(SlotHeader* ptr) : IFieldScanner(ptr), cnt_(0) {}

  /**
   * Prepare memory space.
   *
   * @note obj is a single timestep of sample
   */
  virtual void prepare(Argument& argument, PyObject* obj) { ++cnt_; }

  virtual void finishPrepare(Argument& argument) {
    IVector::resizeOrCreate(argument.ids, cnt_, false);
    cnt_ = 0;
  }

  /**
   * Fill one index to argument.
   */
  virtual void fill(Argument& argument, PyObject* obj) {
    bool ok;
    argument.ids->getData()[cnt_++] = py::castInt<int>(obj, &ok);
    CHECK(ok) << "Cannot cast int " << py::repr(obj);
  }

 private:
  size_t cnt_;
};

class SparseNonValueScanner : public IFieldScanner {
 public:
  explicit SparseNonValueScanner(SlotHeader* ptr)
      : IFieldScanner(ptr), nnz_(0), height_(0) {}

  /**
   * Prepare memory space
   * @note obj is a timestep of one sample.
   */
  virtual void prepare(Argument& argument, PyObject* obj) {
    ++height_;
    nnz_ += py::SequenceHelper(obj).size();
  }

  virtual void finishPrepare(Argument& argument) {
    Matrix::resizeOrCreateSparseMatrix(
        argument.value, height_, headerPtr_->dim, nnz_, NO_VALUE);
  }

  virtual void startFill(Argument& argument) {
    auto smat = (CpuSparseMatrix*)(argument.value.get());
    smat->getRows()[0] = 0;
    nnz_ = 0;
    height_ = 1;
  }

  /**
   * Fill one sparse vector to argument.
   * @note obj is a timestep of one sample.
   */
  virtual void fill(Argument& argument, PyObject* obj) {
    py::SequenceHelper s(obj);
    auto sz = s.size();
    auto smat = (CpuSparseMatrix*)(argument.value.get());
    int* row = smat->getRows();
    int* col = smat->getCols();
    real* dat = smat->getData();
    row[height_] = row[height_ - 1] + (int)sz;

    for (decltype(sz) i = 0; i < sz; ++i) {
      setData(col + nnz_, dat + nnz_, s[i]);
      ++nnz_;
    }
    ++height_;
  }

 protected:
  /**
   * Set a single sparse index and value.
   * @param [out] col sparse index
   * @param [out] dat sparse value
   * @param [in] obj Python Object. For sparse_non_value is a PyInt or PyLong.
   *                 For sparse_value is a Tuple (int, float).
   */
  virtual void setData(int* col, real* dat, PyObject* obj) {
    bool ok;
    *col = py::castInt<int>(obj, &ok);
    CHECK(ok);
  }

  size_t nnz_;
  size_t height_;
};

class SparseValueScanner : public SparseNonValueScanner {
 public:
  explicit SparseValueScanner(SlotHeader* ptr) : SparseNonValueScanner(ptr) {}

  virtual void finishPrepare(Argument& argument) {
    Matrix::resizeOrCreateSparseMatrix(
        argument.value, height_, headerPtr_->dim, nnz_, FLOAT_VALUE);
  }

 protected:
  virtual void setData(int* col, real* dat, PyObject* obj) {
    py::SequenceHelper s(obj);
    SparseNonValueScanner::setData(col, dat, s[0]);
    *dat = (real)s.getDouble(1);
  }
};

/**
 * Sequence Scanner. Scanner for sequence or sub-sequence.
 */
class SequenceScanner : public IFieldScanner {
 public:
  /**
   * Ctor
   * @param innerScanner inner scanner for each timestep or sub-sequence.
   * @param getSeqStartPos A callback, (Argument) => ICpuGpuVectorPtr.
   *                       return a sequence start position or a sub-sequence
   *                       start position.
   */
  SequenceScanner(
      std::unique_ptr<IFieldScanner>&& innerScanner,
      const std::function<ICpuGpuVectorPtr&(Argument&)>& getSeqStartPos)
      : IFieldScanner(nullptr),
        inner_(std::move(innerScanner)),
        cnt_(0),
        getSeqStartPos_(getSeqStartPos) {}

  /**
   * Start prepare. Invoke inner->startPrepare too.
   */
  virtual void startPrepare(Argument& argument) {
    inner_->startPrepare(argument);
  }

  /**
   * Prepare. obj is a list or tuple. it will invoke inner_->prepare for each
   * element of sequence obj.
   */
  virtual void prepare(Argument& argument, PyObject* obj) {
    py::SequenceHelper s(obj);
    ++cnt_;
    for (size_t i = 0; i < s.size(); ++i) {
      inner_->prepare(argument, s[i]);
    }
  }

  /**
   * Finish prepare. invoke inner_->finishPrepare too.
   */
  virtual void finishPrepare(Argument& argument) {
    ICpuGpuVector::resizeOrCreate(getSeqStartPos_(argument), cnt_ + 1, false);
    inner_->finishPrepare(argument);
  }

  /**
   * Start fill. invoke inner->startFill too.
   */
  virtual void startFill(Argument& argument) {
    getSeqStartPos_(argument)->getMutableData(false)[0] = 0;
    cnt_ = 1;
    inner_->startFill(argument);
  }

  /**
   * Fill. Obj is a tuple or list. invoke inner->fill for each element of
   * sequence obj. And set seqStartPos at same time. The seqStartPos will be
   * calculated by getSeqStartPos callback passed in ctor.
   */
  virtual void fill(Argument& argument, PyObject* obj) {
    getSeqStartPos_(argument)->getMutableData(false)[cnt_] =
        getSeqStartPos_(argument)->getMutableData(false)[cnt_ - 1] +
        (int)getSize(obj);
    py::SequenceHelper s(obj);
    ++cnt_;
    for (size_t i = 0; i < s.size(); ++i) {
      inner_->fill(argument, s[i]);
    }
  }

  /**
   * Finish fill. will invoke inner->finishFill too.
   */
  virtual void finishFill(Argument& argument) { inner_->finishFill(argument); }

 protected:
  size_t getSize(PyObject* obj) {
    py::SequenceHelper s(obj);
    auto sc = dynamic_cast<SequenceScanner*>(inner_.get());
    if (sc) {
      size_t sum = 0;
      for (size_t i = 0; i < s.size(); ++i) {
        sum += sc->getSize(s[i]);
      }
      return sum;
    } else {
      return s.size();
    }
  }

 private:
  std::unique_ptr<IFieldScanner> inner_;
  size_t cnt_;
  std::function<ICpuGpuVectorPtr&(Argument&)> getSeqStartPos_;
};

IFieldScanner* IFieldScanner::create(SlotHeader* header) {
  IFieldScanner* retv = nullptr;
  switch (header->slotType) {
    case ST_DENSE:
      retv = new DenseScanner(header);
      break;
    case ST_INDEX:
      retv = new IndexScanner(header);
      break;
    case ST_NON_SPARSE_VALUE:
      retv = new SparseNonValueScanner(header);
      break;
    case ST_SPARSE_VALUE:
      retv = new SparseValueScanner(header);
      break;
    default:
      LOG(FATAL) << "Not implemented " << header->slotType;
  }

  switch (header->seqType) {
    case SQT_NONE:
      break;
    case SQT_SUBSEQ:
      retv = new SequenceScanner(std::unique_ptr<IFieldScanner>(retv),
                                 [](Argument& arg) -> ICpuGpuVectorPtr& {
                                   return arg.subSequenceStartPositions;
                                 });
    // fall through, not break;
    case SQT_SEQ:
      retv = new SequenceScanner(std::unique_ptr<IFieldScanner>(retv),
                                 [](Argument& arg) -> ICpuGpuVectorPtr& {
                                   return arg.sequenceStartPositions;
                                 });
      break;
    default:
      LOG(FATAL) << "Not implemented";
  }

  return retv;
}

/**
 * No Cache Strategy. Will destruct old data immediately and load data from
 * python every pass.
 */
class NoCacheStrategy : public IPyDataProviderCache {
 public:
  virtual bool reset() { return true; }

  virtual void drop(std::deque<PyObjectPtr>* data) { data->clear(); }

  virtual std::deque<PyObjectPtr>* load() { return nullptr; }
};

/**
 * Cache One Pass In Memory strategy.
 *
 * In first pass, will load data from python and store them in memory.
 * The rest passes, will load data from memory.
 */
class CacheOnePassInMemory : public IPyDataProviderCache {
 public:
  CacheOnePassInMemory()
      : objPool_(new std::deque<PyObjectPtr>()),
        droppedPool_(new std::deque<PyObjectPtr>()) {}

  virtual bool reset() {
    if (objPool_->empty() && droppedPool_->empty()) {
      return true;
    } else if (objPool_->empty()) {
      std::swap(objPool_, droppedPool_);
      return false;
    } else {
      LOG(FATAL) << "Unexpected branch";
    }
  }

  virtual void drop(std::deque<PyObjectPtr>* data) {
    size_t orgSize = droppedPool_->size();
    droppedPool_->resize(orgSize + data->size());
    for (size_t i = 0; i < data->size(); ++i) {
      std::swap((*droppedPool_)[orgSize + i], (*data)[i]);
    }
    data->clear();
  }

  virtual std::deque<PyObjectPtr>* load() { return objPool_.get(); }

 private:
  std::unique_ptr<std::deque<PyObjectPtr>> objPool_;
  std::unique_ptr<std::deque<PyObjectPtr>> droppedPool_;
};

IPyDataProviderCache* IPyDataProviderCache::create(CacheType ct) {
  switch (ct) {
    case NO_CACHE:
      return new NoCacheStrategy();
    case CACHE_PASS_IN_MEM:
      return new CacheOnePassInMemory();
    default:
      LOG(FATAL) << "Not implemented";
  }
}
}  // namespace paddle

#endif
