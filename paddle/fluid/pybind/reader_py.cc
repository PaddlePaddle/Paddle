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
#include <string>
#include <vector>
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/operators/reader/buffered_reader.h"
#include "paddle/fluid/operators/reader/compose_reader.h"
#include "paddle/fluid/operators/reader/py_reader.h"
#include "paddle/fluid/platform/place.h"
#include "pybind11/stl.h"

namespace paddle {
namespace pybind {

class FeedReader {
  using ResultDictList =
      std::vector<std::unordered_map<std::string, framework::LoDTensor>>;

 public:
  FeedReader(std::unique_ptr<framework::ReaderHolder> reader,
             const std::vector<std::string> &names, size_t num_places,
             bool drop_last = true)
      : reader_(std::move(reader)),
        names_(names),
        num_places_(num_places),
        drop_last_(drop_last) {}

  ResultDictList ReadNext() {
    std::vector<framework::LoDTensor> tensors;
    reader_->ReadNext(&tensors);
    if (tensors.empty()) return ResultDictList();

    PADDLE_ENFORCE(tensors.size() % names_.size() == 0,
                   "Tensor size: %d, names size: %d", tensors.size(),
                   names_.size());

    size_t read_place_num = tensors.size() / names_.size();

    if (drop_last_ && read_place_num != num_places_) {
      return ResultDictList();
    }

    ResultDictList ret(read_place_num);
    for (size_t i = 0; i < tensors.size(); ++i) {
      ret[i / names_.size()].emplace(names_[i % names_.size()],
                                     std::move(tensors[i]));
    }
    return ret;
  }

  void Start() { reader_->Start(); }

  void Reset() { reader_->ResetAll(); }

 private:
  std::unique_ptr<framework::ReaderHolder> reader_;
  std::vector<std::string> names_;
  size_t num_places_;
  bool drop_last_;
};

static std::unique_ptr<framework::ReaderHolder> CreatePyReader(
    const std::vector<
        std::shared_ptr<operators::reader::LoDTensorBlockingQueue>> &queues,
    const std::vector<platform::Place> &dst_places) {
  std::shared_ptr<framework::ReaderBase> reader;
  if (queues.size() == 1) {
    reader.reset(new operators::reader::PyReader(queues[0]));
  } else {
    reader.reset(new operators::reader::MultiQueuePyReader(queues));
  }
  std::vector<std::shared_ptr<framework::ReaderBase>> buffered_reader;
  buffered_reader.reserve(dst_places.size());
  for (auto &p : dst_places) {
    buffered_reader.emplace_back(
        framework::MakeDecoratedReader<operators::reader::BufferedReader>(
            reader, p, 2));
  }
  reader = framework::MakeDecoratedReader<operators::reader::ComposeReader>(
      buffered_reader);

  auto *holder = new framework::ReaderHolder();
  holder->Reset(reader);
  return std::unique_ptr<framework::ReaderHolder>(holder);
}

namespace py = pybind11;

void BindReader(py::module *module) {
  auto &m = *module;

  namespace reader = ::paddle::operators::reader;

  py::class_<framework::ReaderHolder>(m, "Reader", "")
      .def("start", &framework::ReaderHolder::Start)
      .def("reset", &framework::ReaderHolder::ResetAll);

  py::class_<FeedReader>(m, "FeedReader", "")
      .def("read_next", &FeedReader::ReadNext,
           py::call_guard<py::gil_scoped_release>())
      .def("start", &FeedReader::Start,
           py::call_guard<py::gil_scoped_release>())
      .def("reset", &FeedReader::Reset,
           py::call_guard<py::gil_scoped_release>());

  m.def("create_py_reader",
        [](const std::vector<
               std::shared_ptr<operators::reader::LoDTensorBlockingQueue>>
               queues,
           const std::vector<std::string> &names,
           const std::vector<platform::Place> &dst_places, bool drop_last) {
          return new FeedReader(CreatePyReader(queues, dst_places), names,
                                dst_places.size(), drop_last);
        },
        py::return_value_policy::take_ownership);
}

}  // namespace pybind
}  // namespace paddle
