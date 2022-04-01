// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/data/batch_decode_random_crop_op.h"

namespace paddle {
namespace operators {
namespace data {

class BatchDecodeRandomCropOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_GE(ctx->Inputs("X").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "Inputs(X) of DecodeJpeg should not be empty."));
    PADDLE_ENFORCE_GE(ctx->Outputs("Out").size(), 1UL,
                      platform::errors::InvalidArgument(
                          "Outputs(Out) of DecodeJpeg should not be empty."));
    auto aspect_ratio_min = ctx->Attrs().Get<float>("aspect_ratio_min");
    auto aspect_ratio_max = ctx->Attrs().Get<float>("aspect_ratio_max");
    PADDLE_ENFORCE_GT(
        aspect_ratio_min, 0.,
        platform::errors::InvalidArgument(
            "aspect_ratio_min should be greater than 0, but received "
            "%f",
            aspect_ratio_min));
    PADDLE_ENFORCE_GT(
        aspect_ratio_max, 0.,
        platform::errors::InvalidArgument(
            "aspect_ratio_max should be greater than 0, but received "
            "%f",
            aspect_ratio_max));
    PADDLE_ENFORCE_GE(
        aspect_ratio_max, aspect_ratio_min,
        platform::errors::InvalidArgument(
            "aspect_ratio_max should be greater than aspect_ratio_min, "
            "but received aspect_ratio_max(%d) < aspect_ratio_min(%d)",
            aspect_ratio_max, aspect_ratio_min));

    auto area_min = ctx->Attrs().Get<float>("area_min");
    auto area_max = ctx->Attrs().Get<float>("area_max");
    PADDLE_ENFORCE_GT(area_min, 0.,
                      platform::errors::InvalidArgument(
                          "area_minshould be greater than 0, but received "
                          "%f",
                          area_min));
    PADDLE_ENFORCE_GT(area_max, 0.,
                      platform::errors::InvalidArgument(
                          "area_max should be greater than 0, but received "
                          "%f",
                          area_max));
    PADDLE_ENFORCE_GE(area_max, area_min,
                      platform::errors::InvalidArgument(
                          "area_max should be greater than area_min, "
                          "but received area_max(%f) < area_min(%f)",
                          area_max, area_min));

    auto num_attempts = ctx->Attrs().Get<int64_t>("num_attempts");
    PADDLE_ENFORCE_GT(num_attempts, 0,
                      platform::errors::InvalidArgument(
                          "num_attempts should be a positive integerm, but "
                          "received %d",
                          num_attempts));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(framework::proto::VarType::UINT8,
                                   ctx.GetPlace());
  }

  framework::OpKernelType GetKernelTypeForVar(
      const std::string& var_name, const framework::Tensor& tensor,
      const framework::OpKernelType& expected_kernel_type) const {
    if (var_name == "X") {
      return expected_kernel_type;
    }

    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   tensor.place());
  }
};

class BatchDecodeRandomCropInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    ctx->SetOutputType("Out", framework::proto::VarType::LOD_TENSOR_ARRAY,
                       framework::ALL_ELEMENTS);
  }
};

class BatchDecodeRandomCropOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "A one dimensional uint8 tensor containing the raw bytes "
             "of the JPEG image. It is a tensor with rank 1.")
        .AsDuplicable();
    AddOutput("Out", "The output tensor of DecodeJpeg op").AsDuplicable();
    AddComment(R"DOC(
This operator decodes a JPEG image into a 3 dimensional RGB Tensor 
or 1 dimensional Gray Tensor. Optionally converts the image to the 
desired format. The values of the output tensor are uint8 between 0 
and 255.
)DOC");
    AddAttr<int>("local_rank",
                 "(int64_t)"
                 "The index of the op to start execution");
    AddAttr<int>("num_threads", "Path of the file to be readed.").SetDefault(2);
    AddAttr<int64_t>("host_memory_padding",
                     "(int64, default 0), pinned memory allocation padding "
                     "number for Nvjpeg decoding")
        .SetDefault(0);
    AddAttr<int64_t>("device_memory_padding",
                     "(int64, default 0), device memory allocation padding "
                     "number for Nvjpeg decoding")
        .SetDefault(0);
    AddAttr<std::string>(
        "data_format",
        "(string, default NCHW) Only used in "
        "an optional string from: \"NHWC\", \"NCHW\". "
        "Specify that the data format of the input and output data is "
        "channel_first or channel_last.")
        .SetDefault("NCHW");
    AddAttr<float>("aspect_ratio_min", "").SetDefault(3. / 4.);
    AddAttr<float>("aspect_ratio_max", "").SetDefault(4. / 3.);
    AddAttr<float>("area_min", "").SetDefault(0.08);
    AddAttr<float>("area_max", "").SetDefault(1.);
    AddAttr<int64_t>("num_attempts", "").SetDefault(10);
    AddAttr<int64_t>("program_id",
                     "(int64_t)"
                     "The unique hash id used as cache key for "
                     "decode thread pool");
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    batch_decode_random_crop, ops::data::BatchDecodeRandomCropOp,
    ops::data::BatchDecodeRandomCropOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>)

REGISTER_OP_CPU_KERNEL(batch_decode_random_crop,
                       ops::data::CPUBatchDecodeRandomCropKernel<uint8_t>)
