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

#include <sstream>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/phi/backends/device_memory_aligment.h"
#include "paddle/phi/kernels/funcs/math_function.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/infermeta/multiary.h"

namespace paddle {
namespace operators {

template <typename DeviceContext>
struct FillConstantVisitor {
  FillConstantVisitor(const DeviceContext &dev_ctx,
                      phi::DenseTensor *tensor,
                      const float value,
                      framework::proto::VarType::Type dtype,
                      const framework::ExecutionContext &context)
      : dev_ctx_(dev_ctx),
        tensor_(tensor),
        value_(value),
        dtype_(dtype),
        context_(context) {}

  template <typename T>
  void apply(typename std::enable_if<std::is_same<T, int8_t>::value ||
                                     std::is_same<T, int16_t>::value>::type * =
                 nullptr) const {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Not support data type for set_constant attr"));
  }

  template <typename T>
  void apply(typename std::enable_if<!(std::is_same<T, int8_t>::value ||
                                       std::is_same<T, int16_t>::value)>::type
                 * = nullptr) const {
    phi::funcs::SetConstant<DeviceContext, T> set_constant;
    set_constant(dev_ctx_, tensor_, static_cast<T>(value_));
  }

  const DeviceContext &dev_ctx_;
  phi::DenseTensor *tensor_;
  float value_;
  framework::proto::VarType::Type dtype_;
  const framework::ExecutionContext &context_;
};

template <typename DeviceContext, typename T>
class CoalesceTensorOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto in_var_names = context.InputNames("Input");
    auto out_var_names = context.OutputNames("Output");
    const auto &in_tensors = context.MultiInput<phi::DenseTensor>("Input");
    auto out_tensors = context.MultiOutput<phi::DenseTensor>("Output");

    PADDLE_ENFORCE_GT(in_var_names.size(),
                      static_cast<size_t>(0),
                      platform::errors::InvalidArgument(
                          "The CoalesceTensor operator has no input."));
    PADDLE_ENFORCE_EQ(in_var_names.size(),
                      out_var_names.size(),
                      platform::errors::InvalidArgument(
                          "The number of CoalesceTensor operator's input and "
                          "output is not match, "
                          "input number is %u, output number is %u.",
                          in_var_names.size(),
                          out_var_names.size()));

    // Input & Output check: only support phi::DenseTensor
    bool has_not_init_in_vars = false;
    for (size_t i = 0; i < in_tensors.size(); ++i) {
      PADDLE_ENFORCE_NOT_NULL(
          in_tensors[i],
          platform::errors::InvalidArgument(
              "The %d-th input tensor cannot be nullptr.", i));
      PADDLE_ENFORCE_NOT_NULL(
          out_tensors[i],
          platform::errors::InvalidArgument(
              "The %d-th output tensor cannot be nullptr.", i));
      if (!in_tensors[i]->IsInitialized()) {
        has_not_init_in_vars = true;
      }
    }

    if (has_not_init_in_vars) {
      const auto &concated_shapes =
          context.Attr<std::vector<int64_t>>("concated_shapes");
      const auto &concated_ranks =
          context.Attr<std::vector<int64_t>>("concated_ranks");
      PADDLE_ENFORCE_EQ(concated_ranks.size(),
                        out_tensors.size(),
                        platform::errors::InvalidArgument(
                            "The attribute(concated_ranks) length must be "
                            "equal to the output tensor number."));
      int64_t accumulated_ranks = 0;
      for (size_t i = 0; i < in_tensors.size(); ++i) {
        framework::DDim dims(concated_shapes.data() + accumulated_ranks,
                             concated_ranks[i]);
        if (!in_tensors[i]->IsInitialized()) {
          PADDLE_ENFORCE_EQ(
              in_tensors[i],
              out_tensors[i],
              platform::errors::InvalidArgument(
                  "The %d-th output tensor and %d-th input tensor when the "
                  "%d-th input tensor is not initialized.",
                  i,
                  i,
                  i));
          out_tensors[i]->Resize(dims);
        } else {
          PADDLE_ENFORCE_EQ(
              in_tensors[i]->dims(),
              dims,
              platform::errors::InvalidArgument(
                  "The %d-th input tensor shape does not match the "
                  "attribute(concated_shapes) and "
                  "attribute(concated_ranks).",
                  i));
        }
        accumulated_ranks += concated_ranks[i];
        PADDLE_ENFORCE_LE(accumulated_ranks,
                          concated_shapes.size(),
                          platform::errors::InvalidArgument(
                              "The attribute(concated_shapes) and "
                              "attribute(concated_ranks) do not match."));
      }
      PADDLE_ENFORCE_EQ(accumulated_ranks,
                        concated_shapes.size(),
                        platform::errors::InvalidArgument(
                            "The attribute(concated_shapes) and "
                            "attribute(concated_ranks) do not match."));
    }

    bool use_align = context.Attr<bool>("use_align");
    auto align_size = context.Attr<int>("align_size");
    auto size_of_dtype = context.Attr<int>("user_defined_size_of_dtype");

    if (context.Attr<bool>("check_name")) {
      for (size_t i = 0; i < in_var_names.size(); ++i) {
        PADDLE_ENFORCE_EQ(
            in_var_names[i],
            out_var_names[i],
            platform::errors::InvalidArgument(
                "The input and output variable of CoalesceTensor operator is "
                "different, %dth input is %s, %dth output is %s.",
                i,
                in_var_names[i],
                i,
                out_var_names[i]));
      }
    } else {
      // Init the output as input
      for (size_t i = 0; i < in_tensors.size(); ++i) {
        out_tensors[i]->Resize(in_tensors[i]->dims());
      }
    }

    auto &dev_ctx = context.template device_context<DeviceContext>();

    // Get numel and dtype
    size_t numel = 0;
    auto dtype = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));
    if (size_of_dtype == -1) {
      size_of_dtype = framework::SizeOfType(dtype);
    }
    GetMemSizeAndDtype(in_tensors,
                       in_var_names,
                       &numel,
                       size_of_dtype,
                       context.GetPlace(),
                       use_align,
                       align_size);

    // Alloc the continuous space
    auto fused_tensor = context.Output<phi::DenseTensor>("FusedOutput");
    void *fused_tensor_ptr =
        fused_tensor->Resize(phi::make_ddim({static_cast<int64_t>(numel)}))
            .mutable_data(context.GetPlace(),
                          framework::TransToPhiDataType(dtype));
    VLOG(10) << "Fused tensor addr " << fused_tensor_ptr;

    // Init the continuous space
    size_t offset = 0;
    if (context.Attr<bool>("copy_data")) {
      for (size_t i = 0; i < in_var_names.size(); ++i) {
        size_t len = static_cast<size_t>(in_tensors[i]->numel());
        auto sub_tensor = fused_tensor->Slice(
            static_cast<int64_t>(offset), static_cast<int64_t>(offset + len));
        framework::TensorCopy(
            *in_tensors[i], context.GetPlace(), dev_ctx, &sub_tensor);

        offset += use_align ? phi::Alignment(len * size_of_dtype,
                                             context.GetPlace(),
                                             align_size) /
                                  size_of_dtype
                            : len;
      }
    } else if (context.Attr<bool>("set_constant")) {
      framework::VisitDataType(
          dtype,
          FillConstantVisitor<DeviceContext>(dev_ctx,
                                             fused_tensor,
                                             context.Attr<float>("constant"),
                                             dtype,
                                             context));
    } else if (context.Attr<bool>("persist_output")) {
      for (size_t i = 0; i < out_var_names.size(); ++i) {
        size_t len = static_cast<size_t>(out_tensors[i]->numel());
        auto sub_tensor = fused_tensor->Slice(
            static_cast<int64_t>(offset), static_cast<int64_t>(offset + len));
        // some var may not persistable, or persistable var may not init
        if (out_tensors[i]->IsInitialized()) {
          framework::TensorCopy(
              *out_tensors[i], context.GetPlace(), dev_ctx, &sub_tensor);
        }
        offset += use_align ? phi::Alignment(len * size_of_dtype,
                                             context.GetPlace(),
                                             align_size) /
                                  size_of_dtype
                            : len;
      }
    }

    // Make the outputs point to the continuous space.
    offset = 0;
    std::stringstream ss;
    ss << "alloc_space_for_vars: ";

    for (size_t i = 0; i < out_tensors.size(); ++i) {
      size_t len = static_cast<size_t>(out_tensors[i]->numel());
      auto dim = out_tensors[i]->dims();
      VLOG(4) << len << " " << dim << " " << offset;
      out_tensors[i]
          ->ShareDataWith(fused_tensor->Slice(
              static_cast<int64_t>(offset), static_cast<int64_t>(offset + len)))
          .Resize(dim);
      len = use_align
                ? phi::Alignment(
                      len * size_of_dtype, context.GetPlace(), align_size) /
                      size_of_dtype
                : len;
      ss << "output(" << out_var_names[i] << ")  dim:(" << dim << ")"
         << " address: " << out_tensors[i]->data() << " len: " << len << ", ";
      offset += len;
    }
    PADDLE_ENFORCE_EQ(
        (int64_t)offset,
        fused_tensor->numel(),
        platform::errors::InvalidArgument(
            "The alloc_space_for_vars's offset: %s is unequal with "
            "fused_tensor's numel: %s.",
            offset,
            fused_tensor->numel()));
    VLOG(10) << ss.str();
  }

 private:
  void GetMemSizeAndDtype(
      const std::vector<const phi::DenseTensor *> &lod_tensors,
      const std::vector<std::string> var_names,
      size_t *numel,
      const size_t &size_of_dtype,
      const platform::Place &place,
      const bool use_align = true,
      const int align_size = -1) const {
    PADDLE_ENFORCE_EQ(
        lod_tensors.size(),
        var_names.size(),
        platform::errors::InvalidArgument(
            "The number of input tensor and variable does not match, the "
            "number of input tensor is %u, the number of input variable is %u.",
            lod_tensors.size(),
            var_names.size()));
    *numel = 0;
    std::stringstream ss;
    ss << "alloc_space_for_vars: ";
    for (size_t i = 0; i < var_names.size(); ++i) {
      auto size = lod_tensors[i]->numel();
      PADDLE_ENFORCE_GT(
          size,
          0,
          platform::errors::InvalidArgument(
              "The number of tensor `%s`'s elements is 0.", var_names[i]));
      auto len = use_align
                     ? phi::Alignment(static_cast<size_t>(size) * size_of_dtype,
                                      place,
                                      align_size) /
                           size_of_dtype
                     : static_cast<size_t>(size);
      const void *ptr =
          lod_tensors[i]->IsInitialized() ? lod_tensors[i]->data() : nullptr;
      VLOG(4) << size << " " << len;
      ss << "input(" << var_names[i] << ") dim:(" << lod_tensors[i]->dims()
         << ") "
         << " addres:" << ptr << " len: " << len << ", ";
      *numel += len;
    }
    VLOG(10) << ss.str();
  }
};

class CoalesceTensorOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    auto use_align = ctx->Attrs().Get<bool>("use_align");
    auto align_size = ctx->Attrs().Get<int>("align_size");
    auto size_of_dtype = ctx->Attrs().Get<int>("user_defined_size_of_dtype");

    auto dtype = static_cast<framework::proto::VarType::Type>(
        ctx->Attrs().Get<int>("dtype"));
    if (size_of_dtype == -1) {
      size_of_dtype = framework::SizeOfType(dtype);
    }
    if (ctx->IsRuntime()) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      int64_t numel = 0;
      auto dims = ctx->GetInputsDim("Input");
      for (const auto &dim : dims) {
        auto size = phi::product(dim);
        auto len = use_align ? phi::Alignment(
                                   static_cast<size_t>(size) * size_of_dtype,
                                   phi::GPUPlace(),
                                   align_size) /
                                   size_of_dtype
                             : static_cast<size_t>(size);
        numel += len;
      }
      ctx->SetOutputDim("FusedOutput", phi::make_ddim({numel}));
      VLOG(4) << "FusedOutput size:" << phi::make_ddim({numel});
#else
      return;
#endif
    } else {
      auto alignment = [](size_t size, size_t align_size) {
        size_t remaining = size % align_size;
        auto aligned_size =
            remaining == 0 ? size : size + (align_size - remaining);
        VLOG(4) << remaining << " " << size << " " << align_size << " "
                << aligned_size;
        return aligned_size;
      };
      VLOG(4) << "align_size: " << align_size;
      if (use_align && align_size > 0) {
        int64_t numel = 0;
        auto dims = ctx->GetInputsDim("Input");
        for (const auto &dim : dims) {
          auto size = phi::product(dim);
          auto len = use_align
                         ? alignment(static_cast<size_t>(size) * size_of_dtype,
                                     align_size) /
                               size_of_dtype
                         : static_cast<size_t>(size);
          numel += len;
        }
        ctx->SetOutputDim("FusedOutput", phi::make_ddim({numel}));
        VLOG(4) << "FusedOutput size:" << phi::make_ddim({numel});
      }
    }
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext &context) const override {
    auto dtype = static_cast<framework::proto::VarType::Type>(
        context.Attr<int>("dtype"));
    return phi::KernelKey(dtype, context.GetPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string &var_name,
      const phi::DenseTensor &tensor,
      const phi::KernelKey &expected_kernel_type) const override {
    return phi::KernelKey(phi::Backend::ALL_BACKEND,
                          tensor.layout(),
                          expected_kernel_type.dtype());
  }
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(coalesce_tensor,
                            CoalesceTensorInferShapeFunctor,
                            PD_INFER_META(phi::CoalesceTensorInferMeta));
