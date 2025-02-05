// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/reader_py.h"

#include <exception>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Python.h"

#include "paddle/common/ddim.h"
#include "paddle/common/flags.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/framework/reader.h"
#include "paddle/phi/core/operators/reader/buffered_reader.h"
#include "paddle/phi/core/operators/reader/dense_tensor_blocking_queue.h"
#include "paddle/phi/core/operators/reader/py_reader.h"
#include "pybind11/stl.h"

COMMON_DECLARE_bool(reader_queue_speed_test_mode);

// disable auto conversion to list in Python
PYBIND11_MAKE_OPAQUE(phi::TensorArray);

namespace paddle::pybind {

namespace py = pybind11;
namespace reader = operators::reader;

static paddle::optional<std::vector<int64_t>> DiffTensorShape(
    const phi::DenseTensor &tensor,
    const std::vector<int64_t> &target_shape,
    size_t num_places) {
  auto tensor_shape = tensor.dims();

  int64_t rank = tensor_shape.size();

  if (UNLIKELY(rank == 0)) {
    if (!target_shape.empty()) {  // Tensor rank = 0 but desc does not match
      return common::vectorize<int64_t>(tensor_shape);
    } else {
      return paddle::none;
    }
  }

  PADDLE_ENFORCE_GE(tensor_shape[0],
                    0,
                    common::errors::InvalidArgument(
                        "Tensor shape at dim 0 must not be less than 0"));

  if (!tensor.lod().empty()) {
    tensor_shape[0] = -1;  // unknown shape
  } else {
    int64_t split_size =
        static_cast<int64_t>((tensor_shape[0] + num_places - 1) / num_places);
    int64_t remainder = static_cast<int64_t>(
        split_size == 0 ? 0 : tensor_shape[0] % split_size);
    tensor_shape[0] = split_size;
    if (target_shape[0] >= 0) {  // need check dim 0
      if (tensor_shape[0] != target_shape[0]) {
        return common::vectorize<int64_t>(tensor_shape);
      }

      if (remainder > 0) {
        tensor_shape[0] = remainder;
        return common::vectorize<int64_t>(tensor_shape);
      }
    }
  }

  for (int64_t idx = 1; idx < rank; ++idx) {
    PADDLE_ENFORCE_GE(
        tensor_shape[idx],
        0,
        common::errors::InvalidArgument(
            "Tensor shape at dim %d must not be less than 0", idx));
    if (target_shape[idx] >= 0 &&
        tensor_shape[static_cast<int>(idx)] != target_shape[idx]) {
      return common::vectorize<int64_t>(tensor_shape);
    }
  }

  return paddle::none;
}

// Check whether the tensor shape matches the VarDesc shape
// Return the different shape if exists
static paddle::optional<std::vector<int64_t>> DiffTensorShapeWithVarDesc(
    const phi::DenseTensor &tensor,
    const framework::VarDesc &var_desc,
    size_t num_places) {
  auto desc_shape = var_desc.GetShape();
  return DiffTensorShape(tensor, desc_shape, num_places);
}

static const std::shared_ptr<reader::DenseTensorBlockingQueue> &GetQueue(
    const std::shared_ptr<reader::DenseTensorBlockingQueue> &queue,
    size_t idx) {
  return queue;
}

static const std::shared_ptr<reader::DenseTensorBlockingQueue> &GetQueue(
    const std::shared_ptr<reader::OrderedMultiDeviceDenseTensorBlockingQueue>
        &queue,
    size_t idx) {
  return queue->GetQueue(idx);
}

template <typename QueueType>
class MultiDeviceFeedReader {
 public:
  using ResultDictList =
      std::vector<std::unordered_map<std::string, phi::DenseTensor>>;
  using ResultList = std::vector<phi::TensorArray>;

  static constexpr bool kKeepOrder =
      std::is_same<QueueType,
                   reader::OrderedMultiDeviceDenseTensorBlockingQueue>::value;

  MultiDeviceFeedReader(
      const std::shared_ptr<QueueType> &queue,
      const std::vector<std::string> &names,
      const std::vector<std::vector<int>> &shapes,
      const std::vector<framework::proto::VarType::Type> &dtypes,
      const std::vector<bool> &need_check_feed,
      const std::vector<phi::Place> &dst_places,
      bool use_double_buffer,
      bool drop_last,
      bool pin_memory = false)
      : queue_(queue),
        names_(names),
        pool_(new ::ThreadPool(dst_places.size())),
        readers_(),
        futures_(),
        exceptions_(),
        ret_(),
        drop_last_(drop_last),
        pin_memory_(pin_memory) {
    std::vector<phi::DDim> dims;
    for (auto &shape : shapes) {
      dims.push_back(common::make_ddim(shape));
    }

    auto first_reader = std::make_shared<reader::PyReader>(
        GetQueue(queue, 0), dims, dtypes, need_check_feed);

    auto create_or_get_reader = [&](size_t idx) {
      if (idx == 0 ||
          std::is_same<QueueType, reader::DenseTensorBlockingQueue>::value) {
        return first_reader;
      } else {
        return std::make_shared<reader::PyReader>(
            GetQueue(queue, idx), dims, dtypes, need_check_feed);
      }
    };

    readers_.reserve(dst_places.size());
    for (size_t i = 0; i < dst_places.size(); ++i) {
      auto &p = dst_places[i];
      auto *holder = new framework::ReaderHolder();
      auto reader = create_or_get_reader(i);
      if (use_double_buffer) {
        VLOG(10) << "Creating " << i << "-th BufferedReader";
        holder->Reset(
            framework::MakeDecoratedReader<operators::reader::BufferedReader>(
                reader, p, 2, pin_memory_));
      } else {
        if (phi::is_gpu_place(p)) {
          PADDLE_THROW(common::errors::PermissionDenied(
              "Place cannot be CUDAPlace when use_double_buffer is False"));
        }
        holder->Reset(reader);
      }
      readers_.emplace_back(holder);
    }

    futures_.resize(dst_places.size());
    ret_.resize(dst_places.size());
    exceptions_.assign(dst_places.size(), nullptr);
    ReadAsync();
  }

  bool DropLast() const { return drop_last_; }

  ResultDictList ReadNext() {
    CheckNextStatus();
    ResultDictList result;
    result.reserve(ret_.size());
    for (auto &item : ret_) {
      if (item.empty()) {
        if (!kKeepOrder) result.emplace_back();
        continue;
      }

      result.emplace_back();
      auto &ret = result.back();
      PADDLE_ENFORCE_EQ(names_.size(),
                        item.size(),
                        common::errors::InvalidArgument(
                            "The sample number of reader's input data and the "
                            "input number of feed list are not equal.\n"
                            "Possible reasons are:\n"
                            "  The generator is decorated by `paddle.batch` "
                            "and configured by `set_batch_generator`, but here "
                            "need to used `set_sample_list_generator`."));
      for (size_t j = 0; j < names_.size(); ++j) {
        ret.emplace(names_[j], std::move(item[j]));
      }
    }
    ReadAsync();
    return result;
  }

  ResultList ReadNextList() {
    CheckNextStatus();
    ResultList result;
    result.reserve(ret_.size());
    for (auto &item : ret_) {
      if (kKeepOrder && item.empty()) continue;
      result.emplace_back(std::move(item));
    }
    ReadAsync();
    return result;
  }

  void Reset() {
    Shutdown();
    Start();
    ReadAsync();
  }

  void Shutdown() {
    for (auto &r : readers_) r->Shutdown();
  }

  ~MultiDeviceFeedReader() {
    queue_->Close();
    pool_.reset();
  }

 private:
  enum Status {
    kSuccess = 0,   // Read next data successfully
    kEOF = 1,       // Reach EOF
    kException = 2  // Exception raises when reading
  };

  Status WaitFutures(std::exception_ptr *e) {
    *e = nullptr;
    size_t success_num = 0;
    for (size_t i = 0; i < futures_.size(); ++i) {
      auto each_status = futures_[i].get();
      if (UNLIKELY(each_status != Status::kSuccess)) {
        if (UNLIKELY(each_status == Status::kException)) {
          PADDLE_ENFORCE_NOT_NULL(
              exceptions_[i],
              common::errors::NotFound("exceptions_[%d] is NULL, but the "
                                       "result status is Status::kException",
                                       i));
          *e = exceptions_[i];
          exceptions_[i] = nullptr;
        }
      } else {
        ++success_num;
      }
    }

    if (UNLIKELY(*e)) {
      return Status::kException;
    }

    if (drop_last_) {
      return success_num == futures_.size() ? Status::kSuccess : Status::kEOF;
    } else {
      return success_num > 0 ? Status::kSuccess : Status::kEOF;
    }
  }

  void Start() {
    for (auto &r : readers_) r->Start();
  }

  void ReadAsync() {
    for (size_t i = 0; i < readers_.size(); ++i) {
      futures_[i] = pool_->enqueue([this, i] {
        try {
          readers_[i]->ReadNext(&ret_[i]);
          return ret_[i].empty() ? Status::kEOF : Status::kSuccess;
        } catch (...) {
          exceptions_[i] = std::current_exception();
          return Status::kException;
        }
      });
    }
  }

  void CheckNextStatus() {
    std::exception_ptr e;
    Status status = WaitFutures(&e);

    if (UNLIKELY(e)) {
      PADDLE_ENFORCE_EQ(status,
                        Status::kException,
                        common::errors::NotFound(
                            "The exception raised is not NULL, but "
                            "the result status is not Status::kException"));
      std::rethrow_exception(e);
    }

    if (UNLIKELY(status == Status::kEOF)) {
      VLOG(2) << "Raise StopIteration Exception in Python";
      py::gil_scoped_acquire guard;
      throw py::stop_iteration();
    }

    PADDLE_ENFORCE_EQ(
        status,
        Status::kSuccess,
        common::errors::NotFound("The function executed successfully, but "
                                 "the result status is not Status::kSuccess"));
  }

  std::shared_ptr<QueueType> queue_;
  std::vector<std::string> names_;
  std::unique_ptr<::ThreadPool> pool_;

  std::vector<std::unique_ptr<framework::ReaderHolder>> readers_;

  std::vector<std::future<Status>> futures_;
  std::vector<std::exception_ptr> exceptions_;

  std::vector<phi::TensorArray> ret_;
  bool drop_last_;
  bool pin_memory_;
};

template <typename QueueType>
void BindMultiDeviceReader(py::module *module, const char *reader_name) {
  auto &m = *module;

  using ReaderType = MultiDeviceFeedReader<QueueType>;
  py::class_<ReaderType>(m, reader_name, "")
      .def("read_next",
           &ReaderType::ReadNext,
           py::call_guard<py::gil_scoped_release>())
      .def("read_next_list",
           &ReaderType::ReadNextList,
           py::call_guard<py::gil_scoped_release>())
      .def(
          "read_next_var_list",
          [](ReaderType &self) {
            auto result_list = self.ReadNextList();
            auto &tensor_list = result_list[0];
            std::vector<std::shared_ptr<imperative::VarBase>> var_list;
            var_list.reserve(tensor_list.size());
            auto func = [](phi::DenseTensor &dense_tensor) {
              std::string act_name =
                  imperative::GetCurrentTracer()->GenerateUniqueName(
                      "generated_var");
              auto new_var = std::make_shared<imperative::VarBase>(act_name);
              new_var->SetPersistable(false);
              new_var->SetType(framework::proto::VarType::DENSE_TENSOR);
              new_var->SetDataType(
                  framework::TransToProtoVarType(dense_tensor.dtype()));
              auto *tensor =
                  new_var->MutableVar()->GetMutable<phi::DenseTensor>();
              *tensor = std::move(dense_tensor);
              return new_var;
            };
            for (auto &tensor : tensor_list) {
              var_list.emplace_back(func(tensor));
            }
            return var_list;
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "reset", &ReaderType::Reset, py::call_guard<py::gil_scoped_release>())
      .def("shutdown",
           &ReaderType::Shutdown,
           py::call_guard<py::gil_scoped_release>());
}

void BindReader(py::module *module) {
  auto &m = *module;

  m.def("diff_tensor_shape",
        [](const phi::DenseTensor &tensor,
           const framework::VarDesc &var_desc,
           size_t num_places) -> py::object {
          auto diff = DiffTensorShapeWithVarDesc(tensor, var_desc, num_places);
          if (diff) {
            return py::cast(std::move(diff.get()));
          } else {
            return py::cast(nullptr);
          }
        });

  m.def("diff_tensor_shape",
        [](const phi::DenseTensor &tensor,
           const std::vector<int64_t> &target_shape,
           size_t num_places) -> py::object {
          auto diff = DiffTensorShape(tensor, target_shape, num_places);
          if (diff) {
            return py::cast(std::move(diff.get()));
          } else {
            return py::cast(nullptr);
          }
        });

  m.def(
      "init_dense_tensor_blocking_queue",
      [](framework::Variable &var,
         size_t capacity,
         bool is_ordered) -> py::object {
        VLOG(1) << "init_dense_tensor_blocking_queue";
        if (is_ordered) {
          auto *holder = var.GetMutable<
              reader::OrderedMultiDeviceDenseTensorBlockingQueueHolder>();
          holder->InitOnce(capacity, FLAGS_reader_queue_speed_test_mode);
          return py::cast(holder->GetQueue());
        } else {
          auto *holder =
              var.GetMutable<reader::DenseTensorBlockingQueueHolder>();
          holder->InitOnce(capacity, FLAGS_reader_queue_speed_test_mode);
          return py::cast(holder->GetQueue());
        }
      },
      py::return_value_policy::copy);

  py::class_<framework::ReaderHolder>(m, "Reader", "")
      .def("start", &framework::ReaderHolder::Start)
      .def("reset", &framework::ReaderHolder::ResetAll);

  py::class_<reader::DenseTensorBlockingQueue,
             std::shared_ptr<reader::DenseTensorBlockingQueue>>(
      m, "DenseTensorBlockingQueue", "")
      .def(
          "push",
          [](reader::DenseTensorBlockingQueue &self,
             const phi::TensorArray &dense_tensor_vec) {
            return self.Push(dense_tensor_vec);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("size", &reader::DenseTensorBlockingQueue::Size)
      .def("capacity", &reader::DenseTensorBlockingQueue::Cap)
      .def("close", &reader::DenseTensorBlockingQueue::Close)
      .def("kill", &reader::DenseTensorBlockingQueue::Kill)
      .def("wait_for_inited",
           &reader::DenseTensorBlockingQueue::WaitForInited,
           py::call_guard<py::gil_scoped_release>());

  py::class_<
      reader::OrderedMultiDeviceDenseTensorBlockingQueue,
      std::shared_ptr<reader::OrderedMultiDeviceDenseTensorBlockingQueue>>(
      m, "OrderedMultiDeviceDenseTensorBlockingQueue", "")
      .def(
          "push",
          [](reader::OrderedMultiDeviceDenseTensorBlockingQueue &self,
             const phi::TensorArray &dense_tensor_vec) {
            return self.Push(dense_tensor_vec);
          },
          py::call_guard<py::gil_scoped_release>())
      .def("size", &reader::OrderedMultiDeviceDenseTensorBlockingQueue::Size)
      .def("capacity", &reader::OrderedMultiDeviceDenseTensorBlockingQueue::Cap)
      .def("close", &reader::OrderedMultiDeviceDenseTensorBlockingQueue::Close)
      .def("kill", &reader::OrderedMultiDeviceDenseTensorBlockingQueue::Kill)
      .def("wait_for_inited",
           &reader::OrderedMultiDeviceDenseTensorBlockingQueue::WaitForInited,
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &reader::OrderedMultiDeviceDenseTensorBlockingQueue::Reset);

  BindMultiDeviceReader<reader::DenseTensorBlockingQueue>(
      module, "MultiDeviceFeedReader");
  BindMultiDeviceReader<reader::OrderedMultiDeviceDenseTensorBlockingQueue>(
      module, "OrderedMultiDeviceFeedReader");

  m.def(
      "create_py_reader",
      [](const std::shared_ptr<reader::DenseTensorBlockingQueue> &queue,
         const std::vector<std::string> &names,
         const std::vector<std::vector<int>> &shapes,
         const std::vector<framework::proto::VarType::Type> &dtypes,
         const std::vector<bool> &need_check_feed,
         const std::vector<phi::Place> &dst_places,
         bool use_double_buffer,
         bool drop_last,
         bool pin_memory) {
        return new MultiDeviceFeedReader<reader::DenseTensorBlockingQueue>(
            queue,
            names,
            shapes,
            dtypes,
            need_check_feed,
            dst_places,
            use_double_buffer,
            drop_last,
            pin_memory);
      },
      py::return_value_policy::take_ownership);

  m.def(
      "create_py_reader",
      [](const std::shared_ptr<
             reader::OrderedMultiDeviceDenseTensorBlockingQueue> &queue,
         const std::vector<std::string> &names,
         const std::vector<std::vector<int>> &shapes,
         const std::vector<framework::proto::VarType::Type> &dtypes,
         const std::vector<bool> &need_check_feed,
         const std::vector<phi::Place> &dst_places,
         bool use_double_buffer,
         bool drop_last,
         bool pin_memory) {
        queue->SetDeviceCount(dst_places.size());
        return new MultiDeviceFeedReader<
            reader::OrderedMultiDeviceDenseTensorBlockingQueue>(
            queue,
            names,
            shapes,
            dtypes,
            need_check_feed,
            dst_places,
            use_double_buffer,
            drop_last,
            pin_memory);
      },
      py::return_value_policy::take_ownership);
}

}  // namespace paddle::pybind
