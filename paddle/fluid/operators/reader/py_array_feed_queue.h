// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <algorithm>
#include <condition_variable>  //NOLINT
#include <memory>
#include <mutex>  // NOLINT
#include <vector>
#include "glog/logging.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/reader/py_blocking_queue.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"
#include "paddle/fluid/pybind/tensor_py.h"

namespace paddle {
namespace operators {
namespace reader {

using PyTuple = ::pybind11::tuple;
using PyArray = ::pybind11::array;

template <typename T>
using PyArrayT = ::pybind11::array_t<T, ::pybind11::array::c_style |
                                            ::pybind11::array::forcecast>;

class PyArrayToTensorVisitor : public boost::static_visitor<void> {
 public:
#define PY_ARRAY_TO_TENSOR_WITH_TYPE(dtype, func_name)                       \
  pybind::func_name(tensor_, static_cast<const PyArrayT<dtype>&>(py_array_), \
                    place)

#define PY_ARRAY_TO_TENSOR(func_name)                  \
  if (IsType<size_t>()) {                              \
    PY_ARRAY_TO_TENSOR_WITH_TYPE(size_t, func_name);   \
  } else if (IsType<int64_t>()) {                      \
    PY_ARRAY_TO_TENSOR_WITH_TYPE(int64_t, func_name);  \
  } else if (IsType<int32_t>()) {                      \
    PY_ARRAY_TO_TENSOR_WITH_TYPE(int32_t, func_name);  \
  } else if (IsType<int16_t>()) {                      \
    PY_ARRAY_TO_TENSOR_WITH_TYPE(int16_t, func_name);  \
  } else if (IsType<uint8_t>()) {                      \
    PY_ARRAY_TO_TENSOR_WITH_TYPE(uint8_t, func_name);  \
  } else if (IsType<float>()) {                        \
    PY_ARRAY_TO_TENSOR_WITH_TYPE(float, func_name);    \
  } else if (IsType<double>()) {                       \
    PY_ARRAY_TO_TENSOR_WITH_TYPE(double, func_name);   \
  } else {                                             \
    PADDLE_THROW("unsupported dtype of python array"); \
  }

  PyArrayToTensorVisitor(const PyArray& py_array, framework::Tensor* tensor)
      : py_array_(py_array), tensor_(tensor) {}

  void operator()(const platform::CPUPlace& place) {
    PY_ARRAY_TO_TENSOR(PyCPUTensorSetFromArray);
  }

  void operator()(const platform::CUDAPlace& place) {
#ifdef PADDLE_WITH_CUDA
    PY_ARRAY_TO_TENSOR(PyCUDATensorSetFromArray);
#else
    PADDLE_THROW("CUDAPlace is not supported in CPU only version");
#endif
  }

  void operator()(const platform::CUDAPinnedPlace& place) {
#ifdef PADDLE_WITH_CUDA
    PY_ARRAY_TO_TENSOR(PyCUDAPinnedTensorSetFromArray);
#else
    PADDLE_THROW("CUDAPinnedPlace is not supported in CPU only version");
#endif
  }

#undef PY_ARRAY_TO_TENSOR
#undef PY_ARRAY_TO_TENSOR_WITH_TYPE

 private:
  template <typename T>
  inline bool IsType() const {
    return ::pybind11::isinstance<PyArrayT<T>>(py_array_);
  }

 private:
  const PyArray& py_array_;
  framework::Tensor* tensor_;
};

class PyArrayFeedQueueHolder;

// PyArrayFeedQueue must be thread-safe
class PyArrayFeedQueue {
  friend class PyArrayFeedQueueHolder;

 private:
  PyArrayFeedQueue(size_t capacity, const std::vector<framework::DDim>& dims,
                   const platform::Place& place)
      : dims_(dims), place_(place) {
    queue_.reset(
        new PyBlockingQueue<std::vector<framework::LoDTensor>>(capacity));
  }

 public:
  ~PyArrayFeedQueue() { Close(); }

  bool Enqueue(const std::vector<PyArray>& py_array_vec) {
    auto lod_tensor_vec = PyArrayVecToLoDTensorVec(py_array_vec);
    VLOG(5) << "Enqueue at address " << reinterpret_cast<void*>(this);
    return queue_->Send(std::move(lod_tensor_vec));
  }

  bool Enqueue(const std::vector<framework::LoDTensor>& tensor_vec) {
    VLOG(5) << "Enqueue at address " << reinterpret_cast<void*>(this);
    return queue_->Send(tensor_vec);
  }

  std::vector<framework::LoDTensor> Dequeue() {
    VLOG(5) << "Dequeue at address " << reinterpret_cast<void*>(this);
    std::vector<framework::LoDTensor> ret;
    return queue_->Receive(&ret) ? ret : std::vector<framework::LoDTensor>();
  }

  inline size_t Size() const { return queue_->Size(); }

  inline size_t Cap() const { return queue_->Cap(); }

  inline bool IsClosed() const { return queue_->IsClosed(); }

  inline void Close() { queue_->Close(); }

 private:
  std::vector<framework::LoDTensor> PyArrayVecToLoDTensorVec(
      const std::vector<PyArray>& py_array_vec) {
    PADDLE_ENFORCE(dims_.size() == py_array_vec.size(),
                   "expected input tensor number %d but found %d", dims_.size(),
                   py_array_vec.size());

    size_t i = 0;
    if (py_array_vec.size() > 1) {
      size_t dim0 = py_array_vec[0].shape()[0];
      for (size_t j = 1; j < py_array_vec.size(); ++j) {
        PADDLE_ENFORCE(dim0 == py_array_vec[j].shape()[0],
                       "0-dim of the %d-th input tensor is %d, but 0-dim of "
                       "the 0-th input tensor is %d",
                       j, py_array_vec[j].shape()[0], dim0);
      }
    }

    std::vector<framework::LoDTensor> lod_tensor_vec;
    lod_tensor_vec.reserve(py_array_vec.size());

    std::for_each(
        py_array_vec.begin(), py_array_vec.end(), [&](const PyArray& py_array) {
          for (int64_t j = 1; j < dims_[i].size(); ++j) {
            PADDLE_ENFORCE(
                dims_[i][j] == static_cast<int64_t>(py_array.shape()[j]),
                "expected %d-dim of %d-th input tensor is %d but found %d", j,
                i, dims_[i][j], py_array.shape()[j]);
          }

          lod_tensor_vec.emplace_back(framework::LoDTensor());
          PyArrayToTensorVisitor visitor(py_array, &(lod_tensor_vec.back()));
          boost::apply_visitor(visitor, place_);
          ++i;
        });
    return lod_tensor_vec;
  }

  std::unique_ptr<PyBlockingQueue<std::vector<framework::LoDTensor>>> queue_;
  std::vector<framework::DDim> dims_;
  platform::Place place_;
};

class PyArrayFeedQueueHolder {
 public:
  PyArrayFeedQueueHolder() {}

  void InitOnce(size_t capacity, const std::vector<framework::DDim>& dims,
                const platform::Place& place) {
    PADDLE_ENFORCE(
        feeder_ == nullptr,
        "PyArrayFeedQueueHolder::InitOnce() can only be called once");
    feeder_.reset(new PyArrayFeedQueue(capacity, dims, place));
  }

  std::shared_ptr<PyArrayFeedQueue> GetFeeder() { return feeder_; }
  const std::shared_ptr<PyArrayFeedQueue>& GetFeeder() const { return feeder_; }

 private:
  std::shared_ptr<PyArrayFeedQueue> feeder_;
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle
