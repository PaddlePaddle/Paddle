/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/op_registry.h"
#include "paddle/platform/place.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/platform/gpu_info.h"
#endif

namespace paddle {
namespace operators {

class GetPlacesOp : public framework::OperatorBase {
 public:
  GetPlacesOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}
  void Run(const framework::Scope &scope,
           const platform::DeviceContext &dev_ctx) const override {
    std::string device_type = Attr<std::string>("device_type");
    auto trainer_count = Attr<int>("trainer_count");

    auto out_var_name = Output("Out");
    auto *out_var = scope.FindVar(out_var_name);
    PADDLE_ENFORCE(out_var != nullptr, "Output variable %s cannot be found",
                   out_var_name);

    auto &places = *(out_var->GetMutable<std::vector<platform::Place>>());
    places.resize(trainer_count);
    if (device_type == "CUDA") {
#ifdef PADDLE_WITH_CUDA
      PADDLE_ENFORCE_LT(trainer_count, GetCUDADeviceCount());
      for (int i = 0; i < trainer_count; i++) {
        places.emplace_back(platform::GPUPlace(i));
      }
#else
      PADDLE_THROW("'GPUPlace' is not supported in CPU only device.");
#endif
    } else if (device_type == "CPU") {
      for (int i = 0; i < trainer_count; i++) {
        places.emplace_back(platform::CPUPlace());
      }
    }
  }
};

class GetPlacesOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  GetPlacesOpProtoMaker(framework::OpProto *proto,
                        framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out", "vector of Place");
    AddAttr<int>("trainer_count", "(int)trainer count").SetDefault(1);
    AddAttr<std::string>("device_type",
                         "(string), deivce type can be \"CPU\" and \"CUDA\"")
        .InEnum({"CPU", "CUDA"});
    AddComment(R"DOC(
Returns a list of places based on flags. The list will be used for parallel execution.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(get_places, ops::GetPlacesOp, ops::GetPlacesOpProtoMaker);
