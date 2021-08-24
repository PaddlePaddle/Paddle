/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
class OpDesc;
class InferShapeContext;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
struct CPUPlace;
struct CUDAPlace;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

static void DataCopy(const framework::LoDTensor &src_item,
                     const std::string &fetch_var_name,
                     framework::LoDTensor *dst_item,
                     const platform::DeviceContext &dev_ctx) {
  if (src_item.IsInitialized() && src_item.numel() > 0) {
#ifdef PADDLE_WITH_MKLDNN
    // Conversion from MKL-DNN to Paddle
    if (src_item.layout() == framework::DataLayout::kMKLDNN) {
      framework::Tensor out;
      // Convert to desired Paddle layout, apart from grads of filter
      // as params are not a subject to paddle's data_format
      framework::innerTransDataLayoutFromMKLDNN(
          src_item.layout(), fetch_var_name == framework::GradVarName("Filter")
                                 ? framework::DataLayout::kNCHW
                                 : paddle::platform::MKLDNNDeviceContext::tls()
                                       .get_cur_paddle_data_layout(),
          src_item, &out, platform::CPUPlace());
      TensorCopy(src_item, platform::CPUPlace(), dev_ctx, dst_item);
    } else {
      if (platform::is_gpu_place(src_item.place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        TensorCopy(src_item, platform::CUDAPinnedPlace(), dev_ctx, dst_item);
#endif
      } else {
        TensorCopy(src_item, platform::CPUPlace(), dst_item);
      }
    }
#else
    if (platform::is_gpu_place(src_item.place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      TensorCopy(src_item, platform::CUDAPinnedPlace(), dev_ctx, dst_item);
#endif
    } else {
      TensorCopy(src_item, platform::CPUPlace(), dst_item);
    }
#endif

  } else {
    // Not copy, if the src tensor is empty.
    dst_item->clear();
    dst_item->Resize({0});
  }
  dst_item->set_lod(src_item.lod());
}

class FetchV2Op : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {}

 protected:
  framework::OpKernelType GetKernelTypeForVar(
      const std::string &var_name, const framework::Tensor &tensor,
      const framework::OpKernelType &expected_kernel_type) const override {
    return framework::OpKernelType(expected_kernel_type.data_type_,
                                   expected_kernel_type.place_,
                                   tensor.layout());
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class FetchV2InferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext *ctx) const override {
    ctx->SyncTypeAndDataType("X", "Out");
  }
};

class FetchV2Kernel {
 public:
  void operator()(const framework::ExecutionContext &ctx) const {
    auto fetch_var_name = ctx.InputName("X");
    auto *fetch_var = ctx.InputVar("X");
    if (fetch_var == nullptr) {
      return;
    }
    PADDLE_ENFORCE_EQ(ctx.HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of memcpy_d2h_op is not found."));
    auto *out_var = ctx.OutputVar("Out");
    // Get dev_ctx from ExecutionContext, it's D2H stream
    auto &dev_ctx = ctx.device_context();

    int col = ctx.Attr<int>("col");
    PADDLE_ENFORCE_GE(
        col, 0, platform::errors::InvalidArgument(
                    "Expected the column index (the attribute 'col' of "
                    "operator 'Fetch') of current fetching variable to be "
                    "no less than 0. But received column index = %d.",
                    col));

    auto *fetch_list = out_var->GetMutable<framework::FetchList>();

    if (static_cast<size_t>(col) >= fetch_list->size()) {
      fetch_list->resize(col + 1);
    }

    if (fetch_var->IsType<framework::LoDTensor>()) {
      auto &src_item = fetch_var->Get<framework::LoDTensor>();
      auto *dst_item = &(BOOST_GET(framework::LoDTensor, fetch_list->at(col)));
      DataCopy(src_item, fetch_var_name, dst_item, dev_ctx);
    } else {
      auto &src_item = fetch_var->Get<framework::LoDTensorArray>();
      framework::LoDTensorArray tmp(src_item.size());
      fetch_list->at(col) = tmp;
      auto &dst_item =
          BOOST_GET(framework::LoDTensorArray, fetch_list->at(col));
      for (size_t i = 0; i < src_item.size(); ++i) {
        DataCopy(src_item[i], fetch_var_name, &dst_item[i], dev_ctx);
      }
    }
  }
};

class FetchV2OpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor) The resulted LoDTensor which is expected to return "
             "to users.");
    AddOutput("Out",
              "(vector<LoDTensor>) A fetching list of LoDTensor which may have "
              "different dimension, shape and data type.");
    AddAttr<int>("col", "(int) The column index of fetching object.");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OPERATOR(
    fetch_v2, ops::FetchV2Op, ops::FetchV2OpProtoMaker,
    ops::FetchV2InferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL_FUNCTOR(fetch_v2, float, ops::FetchV2Kernel, double,
                               ops::FetchV2Kernel, int, ops::FetchV2Kernel,
                               int64_t, ops::FetchV2Kernel, bool,
                               ops::FetchV2Kernel, plat::float16,
                               ops::FetchV2Kernel);

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_ROCM)
REGISTER_OP_CUDA_KERNEL_FUNCTOR(fetch_v2, float, ops::FetchV2Kernel, double,
                                ops::FetchV2Kernel, int, ops::FetchV2Kernel,
                                int64_t, ops::FetchV2Kernel, bool,
                                ops::FetchV2Kernel, plat::float16,
                                ops::FetchV2Kernel);
#endif

#ifdef PADDLE_WITH_ASCEND_CL
REGISTER_OP_NPU_KERNEL_FUNCTOR(fetch_v2, float, ops::FetchV2Kernel, double,
                               ops::FetchV2Kernel, int, ops::FetchV2Kernel,
                               int64_t, ops::FetchV2Kernel, bool,
                               ops::FetchV2Kernel, plat::float16,
                               ops::FetchV2Kernel);
#endif
