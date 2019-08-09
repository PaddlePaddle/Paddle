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
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/operators/reader/buffered_reader.h"
#include "paddle/fluid/operators/reader/py_reader.h"
#include "paddle/fluid/platform/place.h"
#include "pybind11/stl.h"

namespace paddle {
namespace pybind {

class MultiDeviceFeedReader {
 public:
  using ResultDictList =
      std::vector<std::unordered_map<std::string, framework::LoDTensor>>;

  MultiDeviceFeedReader(
      const std::shared_ptr<operators::reader::LoDTensorBlockingQueue> &queue,
      const std::vector<std::string> &names,
      const std::vector<platform::Place> &dst_places, bool use_double_buffer)
      : queue_(queue),
        names_(names),
        pool_(new ::ThreadPool(dst_places.size())) {
    std::shared_ptr<framework::ReaderBase> reader(
        new operators::reader::PyReader(queue));

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
    ReadAsync();
  }

  ResultDictList ReadNext() {
    bool success = WaitFutures();

    if (!success) {
      return {};
    }

    ResultDictList result(ret_.size());
    for (size_t i = 0; i < ret_.size(); ++i) {
      for (size_t j = 0; j < names_.size(); ++j) {
        result[i].emplace(names_[j], std::move(ret_[i][j]));
      }
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
  bool WaitFutures() {
    bool success = true;
    for (auto &f : futures_) {
      success &= f.get();
    }
    return success;
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
        readers_[i]->ReadNext(&ret_[i]);
        return !ret_[i].empty();
      });
    }
  }

  std::shared_ptr<operators::reader::LoDTensorBlockingQueue> queue_;
  std::vector<std::string> names_;
  std::unique_ptr<::ThreadPool> pool_;

  std::vector<std::unique_ptr<framework::ReaderHolder>> readers_;

  std::vector<std::future<bool>> futures_;
  std::vector<std::vector<framework::LoDTensor>> ret_;
};

namespace py = pybind11;

void BindReader(py::module *module) {
  auto &m = *module;

  namespace reader = ::paddle::operators::reader;

  py::class_<framework::ReaderHolder>(m, "Reader", "")
      .def("start", &framework::ReaderHolder::Start)
      .def("reset", &framework::ReaderHolder::ResetAll);

  py::class_<MultiDeviceFeedReader>(m, "MultiDeviceFeedReader", "")
      .def("read_next", &MultiDeviceFeedReader::ReadNext,
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &MultiDeviceFeedReader::Reset,
           py::call_guard<py::gil_scoped_release>());

  m.def("create_py_reader",
        [](const std::shared_ptr<operators::reader::LoDTensorBlockingQueue>
               &queue,
           const std::vector<std::string> &names,
           const std::vector<platform::Place> &dst_places,
           bool use_double_buffer) {
          return new MultiDeviceFeedReader(queue, names, dst_places,
                                           use_double_buffer);
        },
        py::return_value_policy::take_ownership);
}

}  // namespace pybind
}  // namespace paddle
