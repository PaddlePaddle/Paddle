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

#include <thread>  // NOLINT
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/reader/queue_based_reader.h"

namespace paddle {
namespace operators {
namespace reader {

class CreateMultiDeviceLoDTensorBlockingQueueOp
    : public framework::OperatorBase {
 public:
  using framework::OperatorBase::OperatorBase;

 protected:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    int dev_cnt = Attr<int>("dev_cnt");
    if (dev_cnt < 0) {
      if (platform::is_gpu_place(place)) {
        dev_cnt = static_cast<int>(platform::GetSelectedDevices().size());
      } else {
        dev_cnt = static_cast<int>(std::thread::hardware_concurrency());
      }
    }

    PADDLE_ENFORCE(dev_cnt > 0, "dev_cnt must be larger than 0");

    auto cap = static_cast<size_t>(Attr<int>("capacity"));
    auto speed_test_mode = Attr<bool>("speed_test_mode");

    auto *out =
        scope.FindVar(Output("Out"))
            ->GetMutable<
                std::shared_ptr<MultiDeviceLoDTensorBlockingQueueHolder>>();
    out->reset(new MultiDeviceLoDTensorBlockingQueueHolder(dev_cnt, cap,
                                                           speed_test_mode));
  }
};

class CreateMultiDeviceLoDTensorBlockingQueueOpInferShape
    : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {}
};

class CreateMultiDeviceLoDTensorBlockingQueueOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "Output queue");

    AddAttr<int>("dev_cnt", "Device number").SetDefault(-1);

    AddAttr<int>("capacity", "Capacity of the queue.");

    AddAttr<bool>("speed_test_mode", "Speed test mode.").SetDefault(false);

    AddComment(R"DOC(
     Create a queue for multi-devices data feeding.
    )DOC");
  }
};

}  // namespace reader
}  // namespace operators
}  // namespace paddle

namespace reader = ::paddle::operators::reader;

REGISTER_OPERATOR(create_multi_device_lod_tensor_blocking_queue,
                  reader::CreateMultiDeviceLoDTensorBlockingQueueOp,
                  reader::CreateMultiDeviceLoDTensorBlockingQueueOpMaker,
                  reader::CreateMultiDeviceLoDTensorBlockingQueueOpInferShape);
