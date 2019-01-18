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

#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/operators/reader/reader_op_registry.h"
#include "paddle/fluid/string/split.h"

DECLARE_string(selected_gpus);

namespace paddle {
namespace operators {
namespace reader {

static int GetNumPlaces(const platform::Place& place) {
  size_t num_places = 0;
  if (platform::is_cpu_place(place)) {
    num_places = getenv("CPU_NUM") == NULL ? std::thread::hardware_concurrency()
                                           : atoi(getenv("CPU_NUM"));
  } else if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    num_places = FLAGS_selected_gpus.empty() == true
                     ? platform::GetCUDADeviceCount()
                     : paddle::string::Split(FLAGS_selected_gpus, ',').size();
#else
    PADDLE_THROW("PADDLE should been compiled with WITH_CUDA=1.");
#endif
  } else {
    PADDLE_ENFORCE("The place of pyreader should be CPU or GPU.");
  }
  return num_places;
}

class PyReader : public framework::FileReader {
 public:
  explicit PyReader(const std::shared_ptr<LoDTensorBlockingQueue>& queue)
      : framework::FileReader() {
    PADDLE_ENFORCE(queue != nullptr, "LoDTensorBlockingQueue must not be null");
    queue_ = queue;
  }

  void ReadNext(std::vector<framework::LoDTensor>* out,
                int dev_id = 0) override {
    bool success;
    *out = queue_->Pop(&success, dev_id);
    if (!success) out->clear();
  }

  ~PyReader() { queue_->Close(); }

  void Shutdown() override { queue_->Close(); }

  void Start() override { queue_->ReOpen(); }

 private:
  std::shared_ptr<LoDTensorBlockingQueue> queue_;
};

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
    PADDLE_ENFORCE_NOT_NULL(
        queue_holder_var,
        "No LoDTensorBlockingQueueHolder variable with name %s found",
        queue_name);
    auto* queue_holder =
        queue_holder_var->template GetMutable<LoDTensorBlockingQueueHolder>();
    size_t num_places = static_cast<size_t>(Attr<int>("num_places"));
    // TODO(Yancey1989): Need more benchmark to enable multiple queue for CPU
    // training
    if (platform::is_gpu_place(dev_place)) {
      if (num_places == 0) num_places = GetNumPlaces(dev_place);
      queue_holder->GetQueue()->ReInitWithMultiDev(num_places);
    }
    out->Reset(std::make_shared<PyReader>(queue_holder->GetQueue()));
  }
};

class CreatePyReaderOpMaker : public FileReaderMakerBase {
 protected:
  void Apply() override {
    AddInput("blocking_queue",
             "Name of the `LoDTensorBlockingQueueHolder` variable");
    AddAttr<int>(
        "num_places",
        "The number of places which used in Executor or ParallelExecutor.")
        .SetDefault(1);
    AddComment(R"DOC(
      Create PyReader to support LoDTensor data feeding in Python side.
      )DOC");
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace reader = ::paddle::operators::reader;

REGISTER_FILE_READER_OPERATOR(create_py_reader, reader::CreatePyReaderOp,
                              reader::CreatePyReaderOpMaker);
