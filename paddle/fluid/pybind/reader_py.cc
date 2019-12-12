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
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/operators/reader/buffered_reader.h"
#include "paddle/fluid/operators/reader/py_reader.h"
#include "paddle/fluid/platform/place.h"
#include "pybind11/stl.h"

namespace paddle {
namespace pybind {

namespace py = pybind11;

class MultiDeviceFeedReader {
 public:
  using ResultDictList =
      std::vector<std::unordered_map<std::string, framework::LoDTensor>>;
  using ResultList = std::vector<std::vector<framework::LoDTensor>>;

  MultiDeviceFeedReader(
      const std::shared_ptr<operators::reader::LoDTensorBlockingQueue> &queue,
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
    std::shared_ptr<framework::ReaderBase> reader(
        new operators::reader::PyReader(queue, dims, dtypes, need_check_feed));

    readers_.reserve(dst_places.size());
    for (auto &p : dst_places) {
      auto *holder = new framework::ReaderHolder();
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

  std::shared_ptr<operators::reader::LoDTensorBlockingQueue> queue_;
  std::vector<std::string> names_;
  std::unique_ptr<::ThreadPool> pool_;

  std::vector<std::unique_ptr<framework::ReaderHolder>> readers_;

  std::vector<std::future<Status>> futures_;
  std::vector<std::exception_ptr> exceptions_;

  std::vector<std::vector<framework::LoDTensor>> ret_;
};

void BindReader(py::module *module) {
  auto &m = *module;

  namespace reader = ::paddle::operators::reader;

  py::class_<framework::ReaderHolder>(m, "Reader", "")
      .def("start", &framework::ReaderHolder::Start)
      .def("reset", &framework::ReaderHolder::ResetAll);

  py::class_<MultiDeviceFeedReader>(m, "MultiDeviceFeedReader", "")
      .def("read_next", &MultiDeviceFeedReader::ReadNext,
           py::call_guard<py::gil_scoped_release>())
      .def("read_next_list", &MultiDeviceFeedReader::ReadNextList,
           py::call_guard<py::gil_scoped_release>())
      .def("read_next_var_list",
           [](MultiDeviceFeedReader &self) {
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
      .def("reset", &MultiDeviceFeedReader::Reset,
           py::call_guard<py::gil_scoped_release>());

  m.def("create_py_reader",
        [](const std::shared_ptr<operators::reader::LoDTensorBlockingQueue>
               &queue,
           const std::vector<std::string> &names,
           const std::vector<std::vector<int>> &shapes,
           const std::vector<framework::proto::VarType::Type> &dtypes,
           const std::vector<bool> &need_check_feed,
           const std::vector<platform::Place> &dst_places,
           bool use_double_buffer) {
          return new MultiDeviceFeedReader(queue, names, shapes, dtypes,
                                           need_check_feed, dst_places,
                                           use_double_buffer);
        },
        py::return_value_policy::take_ownership);
}

}  // namespace pybind
}  // namespace paddle
