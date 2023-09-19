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

#include <string>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/phi/kernels/cast_kernel.h"

namespace paddle {
namespace operators {

template <typename T, typename Context>
void LoadKernel(const Context& dev_ctx,
                const std::string& file_path,
                int64_t seek,
                const std::vector<int64_t>& shape,
                bool load_as_fp16,
                phi::DenseTensor* out) {
  // FIXME(yuyang18): We save variable to local file now, but we should change
  // it to save an output stream.
  std::ifstream fin(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fin),
                    true,
                    phi::errors::Unavailable(
                        "Load operator fail to open file %s, please check "
                        "whether the model file is complete or damaged.",
                        file_path));
  PADDLE_ENFORCE_NOT_NULL(out,
                          phi::errors::InvalidArgument(
                              "The variable to be loaded cannot be found."));

  if (seek != -1) {
    PADDLE_ENFORCE_GE(seek,
                      0,
                      phi::errors::InvalidArgument(
                          "seek witn tensor must great than or equal to 0"));
    framework::DeserializeFromStream(fin, out, dev_ctx, seek, shape);
  } else {
    framework::DeserializeFromStream(fin, out, dev_ctx);
  }

  auto in_dtype = out->dtype();
  auto out_dtype = load_as_fp16 ? phi::DataType::FLOAT16 : in_dtype;
  if (in_dtype != out_dtype) {
    phi::CastKernel<T>(dev_ctx, *out, out_dtype, out);
  }
}

template <typename T, typename Context>
void LoadSelectedRowsKernel(const Context& dev_ctx,
                            const std::string& file_path,
                            int64_t seek,
                            const std::vector<int64_t>& shape,
                            bool load_as_fp16,
                            phi::SelectedRows* out) {
  // FIXME(yuyang18): We save variable to local file now, but we should change
  // it to save an output stream.
  std::ifstream fin(file_path, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fin),
                    true,
                    phi::errors::Unavailable(
                        "Load operator fail to open file %s, please check "
                        "whether the model file is complete or damaged.",
                        file_path));
  PADDLE_ENFORCE_NOT_NULL(out,
                          phi::errors::InvalidArgument(
                              "The variable to be loaded cannot be found."));

  framework::DeserializeFromStream(fin, out, dev_ctx);
}

class LoadOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {}

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::FP32, ctx.GetPlace());
  }
};

class LoadOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "The phi::DenseTensor / SelectedRows need to be loaded");
    AddAttr<bool>(
        "load_as_fp16",
        "If true, the tensor will be first loaded and then "
        "converted to float16 data type. Otherwise, the tensor will be "
        "directly loaded without data type conversion. Default is false.")
        .SetDefault(false);
    AddAttr<std::string>("file_path",
                         R"(Variable will be loaded from "file_path")")
        .AddCustomChecker(
            [](const std::string& path) { return !path.empty(); });
    AddAttr<int64_t>("seek", "(int64_t) Starting for load tensor from seek pos")
        .SetDefault(-1);
    AddAttr<std::vector<int64_t>>("shape",
                                  "(vector<int64_t>) The shape of the output")
        .SetDefault({});
    AddComment(
        "Load operator will load a phi::DenseTensor / SelectedRows variable "
        "from "
        "disk "
        "file.");
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(load, ops::LoadOp, ops::LoadOpProtoMaker);

PD_REGISTER_KERNEL(load, CPU, ALL_LAYOUT, ops::LoadKernel, float) {}
PD_REGISTER_KERNEL(
    load_sr, CPU, ALL_LAYOUT, ops::LoadSelectedRowsKernel, float) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(load, GPU, ALL_LAYOUT, ops::LoadKernel, float) {}
PD_REGISTER_KERNEL(
    load_sr, GPU, ALL_LAYOUT, ops::LoadSelectedRowsKernel, float) {}
#endif

#ifdef PADDLE_WITH_XPU
PD_REGISTER_KERNEL(load, XPU, ALL_LAYOUT, ops::LoadKernel, float) {}
#endif
