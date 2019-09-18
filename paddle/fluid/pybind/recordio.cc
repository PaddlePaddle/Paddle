//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/pybind/recordio.h"

#include <fstream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/recordio/writer.h"

namespace paddle {
namespace pybind {

namespace {

class RecordIOWriter {
 public:
  RecordIOWriter(const std::string& filename, recordio::Compressor compressor,
                 size_t max_num_record)
      : closed_(false),
        stream_(filename, std::ios::binary),
        writer_(&stream_, compressor, max_num_record) {}

  void AppendTensor(const framework::LoDTensor& tensor) {
    tensors_.push_back(tensor);
  }

  void CompleteAppendTensor() {
    auto& ctx =
        *platform::DeviceContextPool::Instance().Get(platform::CPUPlace());
    framework::WriteToRecordIO(&writer_, tensors_, ctx);
    tensors_.clear();
  }

  void Close() {
    PADDLE_ENFORCE(tensors_.empty());
    writer_.Flush();
    stream_.close();
    closed_ = true;
  }

  ~RecordIOWriter() {
    if (!closed_) {
      Close();
    }
  }

 private:
  bool closed_;
  std::vector<framework::LoDTensor> tensors_;
  std::ofstream stream_;
  recordio::Writer writer_;
};

}  // namespace

void BindRecordIOWriter(py::module* m) {
  py::class_<RecordIOWriter> writer(*m, "RecordIOWriter", "");
  py::enum_<recordio::Compressor>(writer, "Compressor", "")
      .value("Snappy", recordio::Compressor::kSnappy)
      .value("NoCompress", recordio::Compressor::kNoCompress);

  writer
      .def("__init__",
           [](RecordIOWriter& self, const std::string& filename,
              recordio::Compressor compressor, size_t max_num_record) {
             new (&self) RecordIOWriter(filename, compressor, max_num_record);
           })
      .def("append_tensor", &RecordIOWriter::AppendTensor)
      .def("complete_append_tensor", &RecordIOWriter::CompleteAppendTensor)
      .def("close", &RecordIOWriter::Close);
}

}  // namespace pybind
}  // namespace paddle
