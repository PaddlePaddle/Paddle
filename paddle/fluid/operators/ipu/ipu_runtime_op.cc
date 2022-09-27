//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#ifdef PADDLE_WITH_IPU

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device/ipu/ipu_backend.h"

namespace paddle {
namespace operators {

class IpuRuntimeOp : public framework::OperatorBase {
 public:
  IpuRuntimeOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const {
    auto ipu_backend = platform::ipu::IpuBackend::GetInstance();
    auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
    framework::RuntimeContext runtime_ctx(inputs_, outputs_, scope);
    framework::ExecutionContext ctx(*this, scope, *dev_ctx, runtime_ctx);
    auto inputs = ctx.MultiInput<framework::Tensor>("FeedList");
    auto outputs = ctx.MultiOutput<framework::Tensor>("FetchList");
    auto output_names = ctx.OutputNames("FetchList");
    VLOG(4) << "IpuRuntime Kernel, begin to run graph";
    ipu_backend->Run(inputs, outputs, ctx);

    // post-run
    // resize tensor when tensor.dims() is empty
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto* out = outputs[i];
      if (out->dims().size() == 0) {
        auto sizeof_dtype = framework::DataTypeSize(out->dtype());
        int64_t dim = out->memory_size() / sizeof_dtype;
        out->Resize({dim});
        VLOG(10) << "set ipu_runtime_op output: " << output_names[i]
                 << " dims from () to: "
                 << "(" << dim << ")";
      }
    }
  }
};

class IpuRuntimeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("FeedList", "FeedList of Graph").AsDuplicable();
    AddOutput("FetchList", "FetchList of Graph").AsDuplicable();
    AddComment(R"DOC(
Run graph by PopART runtime.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(ipu_runtime, ops::IpuRuntimeOp, ops::IpuRuntimeOpMaker);

#endif  // PADDLE_WITH_IPU
