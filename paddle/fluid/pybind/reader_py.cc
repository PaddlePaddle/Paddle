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
#include "gflags/gflags.h"
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/operators/reader/buffered_reader.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/operators/reader/py_reader.h"
#include "paddle/fluid/platform/place.h"
#include "pybind11/stl.h"

DEFINE_bool(reader_queue_speed_test_mode, false,
            "If set true, the queue.pop will only get data from queue but not "
            "remove the data from queue for speed testing");

namespace paddle {
namespace pybind {

namespace py = pybind11;
namespace reader = operators::reader;

static const std::shared_ptr<reader::LoDTensorBlockingQueue> &GetQueue(
    const std::shared_ptr<reader::LoDTensorBlockingQueue> &queue, size_t idx) {
  return queue;
}

static const std::shared_ptr<reader::LoDTensorBlockingQueue> &GetQueue(
    const std::shared_ptr<reader::OrderedMultiDeviceLoDTensorBlockingQueue>
        &queue,
    size_t idx) {
  return queue->GetQueue(idx);
}

template <typename QueueType>
class MultiDeviceFeedReader {
 public:
  using ResultDictList =
      std::vector<std::unordered_map<std::string, framework::LoDTensor>>;
  using ResultList = std::vector<std::vector<framework::LoDTensor>>;

  MultiDeviceFeedReader(
      const std::shared_ptr<QueueType> &queue,
      const std::vector<std::string> &names,
      const std::vector<std::vector<int>> &shapes,
      const std::vector<framework::proto::VarType::Type> &dtypes,
      const std::vector<bool> &need_check_feed,
      const std::vector<platform::Place> &dst_places, bool use_double_buffer)
      : queue_(queue),
        names_(names),
        pool_(new ::ThreadPool(dst_places.size())) {
    std::vector<framework::DDim> dims;
    for (auto &shape : shapes) {
      dims.push_back(framework::make_ddim(shape));
    }

    auto first_reader = std::make_shared<reader::PyReader>(
        GetQueue(queue, 0), dims, dtypes, need_check_feed);

    auto create_or_get_reader = [&](size_t idx) {
      if (idx == 0 ||
          std::is_same<QueueType, reader::LoDTensorBlockingQueue>::value) {
        return first_reader;
      } else {
        return std::make_shared<reader::PyReader>(GetQueue(queue, idx), dims,
                                                  dtypes, need_check_feed);
      }
    };

    readers_.reserve(dst_places.size());
    for (size_t i = 0; i < dst_places.size(); ++i) {
      auto &p = dst_places[i];
      auto *holder = new framework::ReaderHolder();
      auto reader = create_or_get_reader(i);
      if (use_double_buffer) {
        holder->Reset(
            framework::MakeDecoratedReader<operators::reader::BufferedReader>(
                reader, p, 2));
      } else {
        if (platform::is_gpu_place(p)) {
          PADDLE_THROW(
              "Place cannot be CUDAPlace when use_double_buffer is False");
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

  ResultDictList ReadNext() {
    CheckNextStatus();
    ResultDictList result(ret_.size());
    for (size_t i = 0; i < ret_.size(); ++i) {
      for (size_t j = 0; j < names_.size(); ++j) {
        result[i].emplace(names_[j], std::move(ret_[i][j]));
      }
    }
    ReadAsync();
    return result;
  }

  ResultList ReadNextList() {
    CheckNextStatus();
    ResultList result;
    result.reserve(ret_.size());
    for (size_t i = 0; i < ret_.size(); ++i) {
      result.emplace_back(std::move(ret_[i]));
    }
    ReadAsync();
    return result;
  }

  void Reset() {
    Shutdown();
    Start();
    ReadAsync();
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

  Status WaitFutures(std::exception_ptr *excep) {
    bool is_success = true;
    *excep = nullptr;
    for (size_t i = 0; i < futures_.size(); ++i) {
      auto each_status = futures_[i].get();
      if (UNLIKELY(each_status != Status::kSuccess)) {
        is_success = false;
        if (UNLIKELY(each_status == Status::kException)) {
          PADDLE_ENFORCE_NOT_NULL(exceptions_[i]);
          *excep = exceptions_[i];
          exceptions_[i] = nullptr;
        }
      }
    }

    if (UNLIKELY(*excep)) {
      return Status::kException;
    } else {
      return is_success ? Status::kSuccess : Status::kEOF;
    }
  }

  void Shutdown() {
    for (auto &r : readers_) r->Shutdown();
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
    std::exception_ptr excep;
    Status status = WaitFutures(&excep);

    if (UNLIKELY(excep)) {
      PADDLE_ENFORCE_EQ(status, Status::kException);
      std::rethrow_exception(excep);
    }

    if (UNLIKELY(status == Status::kEOF)) {
      VLOG(2) << "Raise StopIteration Exception in Python";
      py::gil_scoped_acquire guard;
      throw py::stop_iteration();
    }

    PADDLE_ENFORCE_EQ(status, Status::kSuccess);
  }

  std::shared_ptr<QueueType> queue_;
  std::vector<std::string> names_;
  std::unique_ptr<::ThreadPool> pool_;

  std::vector<std::unique_ptr<framework::ReaderHolder>> readers_;

  std::vector<std::future<Status>> futures_;
  std::vector<std::exception_ptr> exceptions_;

  std::vector<std::vector<framework::LoDTensor>> ret_;
};

template <typename QueueType>
void BindMultiDeviceReader(py::module *module, const char *reader_name) {
  auto &m = *module;

  using ReaderType = MultiDeviceFeedReader<QueueType>;
  py::class_<ReaderType>(m, reader_name, "")
      .def("read_next", &ReaderType::ReadNext,
           py::call_guard<py::gil_scoped_release>())
      .def("read_next_list", &ReaderType::ReadNextList,
           py::call_guard<py::gil_scoped_release>())
      .def("read_next_var_list",
           [](ReaderType &self) {
             auto result_list = self.ReadNextList();
             auto &tensor_list = result_list[0];
             std::vector<std::shared_ptr<imperative::VarBase>> var_list;
             var_list.reserve(tensor_list.size());
             auto func = [](framework::LoDTensor &lod_tensor) {
               std::string act_name =
                   imperative::GetCurrentTracer()->GenerateUniqueName(
                       "generated_var");
               auto new_var = std::make_shared<imperative::VarBase>(act_name);
               new_var->SetPersistable(false);
               new_var->SetType(framework::proto::VarType::LOD_TENSOR);
               new_var->SetDataType(lod_tensor.type());
               auto *tensor =
                   new_var->MutableVar()->GetMutable<framework::LoDTensor>();
               *tensor = std::move(lod_tensor);
               return new_var;
             };
             for (auto &tensor : tensor_list) {
               var_list.emplace_back(func(tensor));
             }
             return var_list;
           },
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &ReaderType::Reset,
           py::call_guard<py::gil_scoped_release>());
}

void BindReader(py::module *module) {
  auto &m = *module;

  m.def("init_lod_tensor_blocking_queue",
        [](framework::Variable &var, size_t capacity,
           bool is_ordered) -> py::object {
          VLOG(1) << "init_lod_tensor_blocking_queue";
          if (is_ordered) {
            auto *holder = var.GetMutable<
                reader::OrderedMultiDeviceLoDTensorBlockingQueueHolder>();
            holder->InitOnce(capacity, FLAGS_reader_queue_speed_test_mode);
            return py::cast(holder->GetQueue());
          } else {
            auto *holder =
                var.GetMutable<reader::LoDTensorBlockingQueueHolder>();
            holder->InitOnce(capacity, FLAGS_reader_queue_speed_test_mode);
            return py::cast(holder->GetQueue());
          }
        },
        py::return_value_policy::copy);

  py::class_<framework::ReaderHolder>(m, "Reader", "")
      .def("start", &framework::ReaderHolder::Start)
      .def("reset", &framework::ReaderHolder::ResetAll);

  py::class_<reader::LoDTensorBlockingQueue,
             std::shared_ptr<reader::LoDTensorBlockingQueue>>(
      m, "LoDTensorBlockingQueue", "")
      .def("push",
           [](reader::LoDTensorBlockingQueue &self,
              const std::vector<framework::LoDTensor> &lod_tensor_vec) {
             return self.Push(lod_tensor_vec);
           },
           py::call_guard<py::gil_scoped_release>())
      .def("size", &reader::LoDTensorBlockingQueue::Size)
      .def("capacity", &reader::LoDTensorBlockingQueue::Cap)
      .def("close", &reader::LoDTensorBlockingQueue::Close)
      .def("kill", &reader::LoDTensorBlockingQueue::Kill)
      .def("wait_for_inited", &reader::LoDTensorBlockingQueue::WaitForInited,
           py::call_guard<py::gil_scoped_release>());

  py::class_<reader::OrderedMultiDeviceLoDTensorBlockingQueue,
             std::shared_ptr<reader::OrderedMultiDeviceLoDTensorBlockingQueue>>(
      m, "OrderedMultiDeviceLoDTensorBlockingQueue", "")
      .def("push",
           [](reader::OrderedMultiDeviceLoDTensorBlockingQueue &self,
              const std::vector<framework::LoDTensor> &lod_tensor_vec) {
             return self.Push(lod_tensor_vec);
           },
           py::call_guard<py::gil_scoped_release>())
      .def("size", &reader::OrderedMultiDeviceLoDTensorBlockingQueue::Size)
      .def("close", &reader::OrderedMultiDeviceLoDTensorBlockingQueue::Close)
      .def("kill", &reader::OrderedMultiDeviceLoDTensorBlockingQueue::Kill)
      .def("wait_for_inited",
           &reader::OrderedMultiDeviceLoDTensorBlockingQueue::WaitForInited,
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &reader::OrderedMultiDeviceLoDTensorBlockingQueue::Reset);

  BindMultiDeviceReader<reader::LoDTensorBlockingQueue>(
      module, "MultiDeviceFeedReader");
  BindMultiDeviceReader<reader::OrderedMultiDeviceLoDTensorBlockingQueue>(
      module, "OrderedMultiDeviceFeedReader");

  m.def("create_py_reader",
        [](const std::shared_ptr<reader::LoDTensorBlockingQueue> &queue,
           const std::vector<std::string> &names,
           const std::vector<std::vector<int>> &shapes,
           const std::vector<framework::proto::VarType::Type> &dtypes,
           const std::vector<bool> &need_check_feed,
           const std::vector<platform::Place> &dst_places,
           bool use_double_buffer) {
          return new MultiDeviceFeedReader<reader::LoDTensorBlockingQueue>(
              queue, names, shapes, dtypes, need_check_feed, dst_places,
              use_double_buffer);
        },
        py::return_value_policy::take_ownership);

  m.def(
      "create_py_reader",
      [](const std::shared_ptr<reader::OrderedMultiDeviceLoDTensorBlockingQueue>
             &queue,
         const std::vector<std::string> &names,
         const std::vector<std::vector<int>> &shapes,
         const std::vector<framework::proto::VarType::Type> &dtypes,
         const std::vector<bool> &need_check_feed,
         const std::vector<platform::Place> &dst_places,
         bool use_double_buffer) {
        queue->SetDeviceCount(dst_places.size());
        return new MultiDeviceFeedReader<
            reader::OrderedMultiDeviceLoDTensorBlockingQueue>(
            queue, names, shapes, dtypes, need_check_feed, dst_places,
            use_double_buffer);
      },
      py::return_value_policy::take_ownership);
}

}  // namespace pybind
}  // namespace paddle
