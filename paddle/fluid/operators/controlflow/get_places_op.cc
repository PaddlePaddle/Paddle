/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
class Scope;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/platform/device/gpu/gpu_info.h"
#endif

namespace paddle {
namespace operators {

static size_t CUDADevCount() {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  return platform::GetGPUDeviceCount();
#else
  return 0UL;
#endif
}

class GetPlacesOp : public framework::OperatorBase {
 public:
  GetPlacesOp(const std::string &type, const framework::VariableNameMap &inputs,
              const framework::VariableNameMap &outputs,
              const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    bool is_gpu;
    if (Attr<std::string>("device_type") == "AUTO") {
      is_gpu = platform::is_gpu_place(place);
    } else {
      is_gpu = Attr<std::string>("device_type") == "CUDA";
    }
    auto device_count = static_cast<size_t>(Attr<int>("device_count"));
    if (device_count == 0) {
      device_count =
          is_gpu ? CUDADevCount() : std::thread::hardware_concurrency();
    }
    PADDLE_ENFORCE_NE(device_count, 0UL, platform::errors::InvalidArgument(
                                             "Cannot indicate %s device count",
                                             is_gpu ? "GPU" : "CPU"));

    auto out_var_name = Output("Out");
    auto &places = *(GET_DATA_SAFELY(scope.FindVar(out_var_name), "Output",
                                     "Out", "GetPlaces")
                         .GetMutable<platform::PlaceList>());
    places.reserve(device_count);
    if (is_gpu) {
      PADDLE_ENFORCE_LE(device_count, CUDADevCount(),
                        platform::errors::InvalidArgument(
                            "Only %d CUDA devices found, cannot set to %d",
                            CUDADevCount(), device_count));
      for (size_t i = 0; i < device_count; ++i) {
        places.emplace_back(platform::CUDAPlace(static_cast<int>(i)));
      }
    } else {
      for (size_t i = 0; i < device_count; ++i) {
        places.emplace_back(platform::CPUPlace());
      }
    }
  }
};

class GetPlacesOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "vector of Place");
    AddAttr<int>("device_count", "device count").SetDefault(0);
    AddAttr<std::string>("device_type", "device type")
        .InEnum({"CUDA", "CPU", "AUTO"})
        .SetDefault("AUTO");
    AddComment(R"DOC(
Returns a list of places based on arguments. The list will be used for parallel
execution.
)DOC");
  }
};

class GetPlacesInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SetOutputType("Out", framework::proto::VarType::PLACE_LIST,
                       framework::ALL_ELEMENTS);
  }
};

class GetPlacesInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    // Do nothing
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(
    get_places, ops::GetPlacesOp, ops::GetPlacesOpProtoMaker,
    ops::GetPlacesInferVarType, ops::GetPlacesInferShape,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
