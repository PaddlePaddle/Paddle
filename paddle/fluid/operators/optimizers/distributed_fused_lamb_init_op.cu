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

#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/operators/math/algorithm.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/optimizers/cast_with_ptr.h"
#include "paddle/fluid/operators/optimizers/distributed_fused_lamb_init_op.h"
#include "paddle/fluid/operators/tensor_to_string.h"
#include "paddle/fluid/platform/device/gpu/gpu_launch_config.h"

namespace paddle {
namespace operators {

struct ParamGradInfo {
  framework::Tensor *param_t{nullptr};
  framework::Tensor *grad_t{nullptr};
  size_t idx{0};
  size_t numel{0};
  size_t numel_with_padding{0};
  size_t numel_offset{0};
};

static std::ostream &operator<<(std::ostream &os, const ParamGradInfo &info) {
  return os << "{Param(" << info.param_t << "),Grad(" << info.grad_t << "),idx("
            << info.idx << "),numel(" << info.numel << "),numel_with_padding("
            << info.numel_with_padding << "),numel_offset(" << info.numel_offset
            << "),padding(" << info.numel_offset + info.numel_with_padding
            << "-" << info.numel_offset + info.numel << "="
            << info.numel_with_padding - info.numel << ")}";
}

struct ParamGradInfoNumelOffsetCompFunctor {
  bool operator()(const ParamGradInfo &x, const ParamGradInfo &y) const {
    return x.numel_offset < y.numel_offset;
  }

  bool operator()(const ParamGradInfo &x, size_t y) const {
    return x.numel_offset < y;
  }

  bool operator()(size_t x, const ParamGradInfo &y) const {
    return x < y.numel_offset;
  }

  bool operator()(size_t x, size_t y) const { return x < y; }
};

static size_t GetAlignSize(size_t n, size_t alignment) {
  auto remainder = n % alignment;
  return remainder == 0 ? n : n + alignment - remainder;
}

// gcd(x, y) = gcd(y, x % y)
// gcd(x, 0) = x
static size_t GCD(size_t x, size_t y) {
  while (y > 0) {
    auto tmp = x;
    x = y;
    y = tmp % y;
  }
  return x;
}

static size_t LCM(size_t x, size_t y) { return x / GCD(x, y) * y; }

// Shard the ParamGradInfo list by the numel size [start_size, end_size)
// The final results should be:
//
// start_size = sum(infos[0:i].numel_with_padding) + start_numel_offset, where
// start_numel_offset <= infos[i].numel_with_padding
//
// end_size = sum(infos[0:j].numel_with_padding) + end_numel_offset, where
// end_numel_offset <= infos[j].numel_with_padding
static void GetParamGradShardInfo(const std::vector<ParamGradInfo> &infos,
                                  size_t start_size, size_t end_size,
                                  size_t *start_idx, size_t *end_idx,
                                  size_t *start_numel_offset,
                                  size_t *end_numel_offset) {
  VLOG(10) << "NumelOffset: "
           << string::join_strings(infos, ",", [](const ParamGradInfo &info) {
                return info.numel_offset;
              });
  VLOG(10) << "start_size = " << start_size << " , end_size = " << end_size;

  if (infos.empty()) {
    PADDLE_ENFORCE_EQ(start_size, 0, platform::errors::InvalidArgument(
                                         "start_size should be 0."));
    PADDLE_ENFORCE_EQ(end_size, 0, platform::errors::InvalidArgument(
                                       "end_size should be 0."));
    *start_idx = 0;
    *end_idx = 0;
    *start_numel_offset = 0;
    *end_numel_offset = 0;
    return;
  }

  PADDLE_ENFORCE_LT(start_size, end_size,
                    platform::errors::InvalidArgument(
                        "start_size should be less than end_size."));
  size_t n = infos.size();
  ParamGradInfoNumelOffsetCompFunctor comp;
  auto i = static_cast<size_t>(
      std::lower_bound(infos.begin(), infos.end(), start_size, comp) -
      infos.begin());
  if (i == n || infos[i].numel_offset != start_size) {
    PADDLE_ENFORCE_GT(
        i, 0, platform::errors::InvalidArgument(
                  "Cannot find suitable sharding which is between [%d, %d)",
                  start_size, end_size));
    --i;
  }
  PADDLE_ENFORCE_LT(
      i, n, platform::errors::InvalidArgument(
                "Cannot find suitable sharding which is between [%d, %d)",
                start_size, end_size));
  *start_idx = i;
  *start_numel_offset = start_size - infos[i].numel_offset;
  auto j = static_cast<size_t>(
      std::lower_bound(infos.begin(), infos.end(), end_size, comp) -
      infos.begin());
  *end_idx = j - 1;
  *end_numel_offset = end_size - infos[j - 1].numel_offset;
  PADDLE_ENFORCE_GT(*end_numel_offset, 0,
                    platform::errors::InvalidArgument(
                        "Internal error when sharding, this may be a bug "
                        "caused by empty parameter."));
  VLOG(10) << "Sharding [start_size=" << start_size << ", end_size=" << end_size
           << "): " << (*start_idx) << ":" << (*start_numel_offset) << " -> "
           << (*end_idx) << ":" << (*end_numel_offset);
}

static size_t FillAlignmentPaddingInfo(std::vector<ParamGradInfo> *infos,
                                       size_t alignment, size_t nranks,
                                       framework::proto::VarType::Type dtype) {
  auto sizeof_dtype = framework::SizeOfType(dtype);
  PADDLE_ENFORCE_EQ(
      alignment % sizeof_dtype, 0,
      platform::errors::InvalidArgument(
          "The attr(alignment) should be exactly divided by sizeof(T) %d.",
          sizeof_dtype));
  alignment /= sizeof_dtype;

  size_t total_numel_sum_with_padding = 0;
  size_t n = infos->size();
  auto lcm = LCM(alignment, nranks);
  for (size_t i = 0; i < n; ++i) {
    auto &info = (*infos)[i];
    size_t numel_with_padding =
        GetAlignSize(info.numel, i + 1 == n ? lcm : alignment);
    info.numel_with_padding = numel_with_padding;
    info.numel_offset = total_numel_sum_with_padding;
    total_numel_sum_with_padding += numel_with_padding;
  }
  return total_numel_sum_with_padding;
}

template <typename T>
static T *TensorFillConstant(const platform::CUDADeviceContext &dev_ctx,
                             framework::Tensor *tensor,
                             const framework::DDim &dims, T value) {
  tensor->Resize(dims);
  auto *ptr = tensor->mutable_data<T>(dev_ctx.GetPlace());
  math::SetConstant<platform::CUDADeviceContext, T> set_constant;
  set_constant(dev_ctx, tensor, value);
  return ptr;
}

static framework::Tensor CastDataForInitedTensor(
    const platform::CUDADeviceContext &dev_ctx, framework::Tensor *origin,
    framework::Tensor *fused_out, size_t numel_offset) {
  PADDLE_ENFORCE_EQ(origin->IsInitialized(), true,
                    platform::errors::InvalidArgument(
                        "The tensor to be cast should be initialized."));

  PADDLE_ENFORCE_EQ(fused_out->type(), framework::proto::VarType::FP32,
                    platform::errors::InvalidArgument(
                        "The dst tensor to be cast should be FP32 tensor."));
  PADDLE_ENFORCE_EQ(origin->type(), framework::proto::VarType::FP16,
                    platform::errors::InvalidArgument(
                        "The src tensor to be cast should be FP16 tensor."));
  auto *dst = fused_out->data<float>() + numel_offset;
  auto *src = origin->data<platform::float16>();
  auto numel = origin->numel();
  LaunchCastKernel(dev_ctx, src, dst, numel);
  VLOG(10) << "Cast from FP32 -> FP16, range: [" << numel_offset << ", "
           << numel_offset + numel << ")"
           << " , total: [0, " << fused_out->numel() << ")";
  framework::DDim fused_out_dim = fused_out->dims();
  auto fused_out_numel = fused_out->numel();
  fused_out->Resize({fused_out_numel});
  auto sliced_tensor = fused_out->Slice(numel_offset, numel + numel_offset);
  fused_out->Resize(fused_out_dim);
  return sliced_tensor;
}

static framework::Tensor CopyAndShareBufferForInitedTensor(
    framework::Tensor *origin, framework::Tensor *fused_out,
    size_t numel_offset, gpuStream_t stream) {
  PADDLE_ENFORCE_EQ(
      origin->IsInitialized(), true,
      platform::errors::InvalidArgument(
          "The tensor to be copied and shared data should be initialized."));
  auto dtype = fused_out->type();
  PADDLE_ENFORCE_EQ(origin->type(), dtype,
                    platform::errors::InvalidArgument(
                        "The tensor to be copied and shared data should be "
                        "have the same data type."));
  auto place = fused_out->place();
  PADDLE_ENFORCE_EQ(
      origin->place(), place,
      platform::errors::InvalidArgument("The tensor to be copied and shared "
                                        "data should be have the same place."));
  PADDLE_ENFORCE_EQ(
      platform::is_gpu_place(place), true,
      platform::errors::InvalidArgument(
          "The tensor to be copied and shared data should be on GPU place."));

  auto numel = origin->numel();
  framework::DDim fused_out_dim = fused_out->dims();
  auto fused_out_numel = fused_out->numel();
  auto sliced_tensor = fused_out->Resize({fused_out_numel})
                           .Slice(numel_offset, numel + numel_offset);
  memory::Copy(place, sliced_tensor.data(), place, origin->data(),
               numel * framework::SizeOfType(dtype), stream);
  origin->ShareBufferWith(sliced_tensor);
  fused_out->Resize(fused_out_dim);
  VLOG(10) << "Copy and share buffer, range: [" << numel_offset << ", "
           << numel_offset + numel << ") , total: [0, " << fused_out->numel()
           << ") , dtype = " << framework::DataTypeToString(dtype);
  return sliced_tensor;
}

static void ShareBufferForNonInitedTensor(framework::Tensor *origin,
                                          framework::Tensor *fused_out,
                                          size_t numel_offset,
                                          const framework::DDim &dims) {
  PADDLE_ENFORCE_EQ(
      origin->IsInitialized(), false,
      platform::errors::InvalidArgument(
          "The tensor to be shared data should not be initialized."));

  framework::DDim fused_out_dim = fused_out->dims();
  auto fused_out_numel = fused_out->numel();
  auto numel = framework::product(dims);
  *origin = fused_out->Resize({fused_out_numel})
                .Slice(numel_offset, numel + numel_offset);
  origin->Resize(dims);
  fused_out->Resize(fused_out_dim);
  VLOG(10) << "Share buffer for non-inited, range: [" << numel_offset << ", "
           << numel_offset + numel << "), total: [0, " << fused_out->numel()
           << ") , dtype = " << framework::DataTypeToString(fused_out->type());
}

template <typename OffsetT, typename IndexT>
static __global__ void LambFillFusedIndicesCUDAKernel(const OffsetT *offsets,
                                                      IndexT *out,
                                                      int offset_num,
                                                      int out_num) {
  CUDA_KERNEL_LOOP_TYPE(i, out_num, int) {
    auto idx = math::LowerBound(offsets, offset_num, i);
    if (idx == offset_num || offsets[idx] != i) {
      --idx;
    }
    out[i] = idx;
  }
}

template <typename T>
class DistributedFusedLambInitOpKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    VLOG(10) << "starts to run DistributedFusedLambInitOp";
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto place = ctx.GetPlace();
    auto stream = dev_ctx.stream();

    // Step 1: Check Input(Param) and Output(ParamOut), Input(Grad) and
    // Output(GradOut)
    auto params = ctx.MultiInput<framework::Tensor>("Param");
    auto grads = ctx.MultiInput<framework::Tensor>("Grad");
    auto master_params = ctx.MultiOutput<framework::Tensor>("MasterParamOut");
    std::vector<ParamGradInfo> fp32_infos, fp16_infos;
    {
      PADDLE_ENFORCE_EQ(params.size(), grads.size(),
                        platform::errors::InvalidArgument(
                            "The parameter number and parameter gradient "
                            "number should be the same."));

      auto params_out = ctx.MultiOutput<framework::Tensor>("ParamOut");
      auto grads_out = ctx.MultiOutput<framework::Tensor>("GradOut");
      PADDLE_ENFORCE_EQ(
          params.size(), params_out.size(),
          platform::errors::InvalidArgument("Input(Param) and Output(ParamOut) "
                                            "should have the same number."));
      PADDLE_ENFORCE_EQ(
          grads.size(), grads_out.size(),
          platform::errors::InvalidArgument(
              "Input(Grad) and Output(GradOut) should have the same number."));
      size_t n = params.size();
      VLOG(10) << "parameter number: " << n;
      for (size_t i = 0; i < n; ++i) {
        auto *p = params[i];
        auto *g = grads[i];
        auto *p_out = params_out[i];
        auto *g_out = grads_out[i];

        PADDLE_ENFORCE_NOT_NULL(
            p, platform::errors::InvalidArgument(
                   "The %d-th parameter should not be nullptr.", i));
        PADDLE_ENFORCE_EQ(p->IsInitialized(), true,
                          platform::errors::InvalidArgument(
                              "The %d-th parameter should be initialized.", i));
        PADDLE_ENFORCE_EQ(
            p->place(), place,
            platform::errors::InvalidArgument(
                "The %d-th parameter is not initialized on the right place.",
                i));
        PADDLE_ENFORCE_EQ(p, p_out,
                          platform::errors::InvalidArgument(
                              "The %d-th Input(Param) and Output(ParamOut) "
                              "should be the same tensor.",
                              i));

        auto dtype = p->type();
        PADDLE_ENFORCE_NOT_NULL(
            g, platform::errors::InvalidArgument(
                   "The %d-th gradient should not be nullptr.", i));
        PADDLE_ENFORCE_EQ(g, g_out,
                          platform::errors::InvalidArgument(
                              "The %d-th Input(Grad) and Output(Grad) should "
                              "be the same tensor."));
        auto numel = p->numel();
        PADDLE_ENFORCE_GT(numel, 0,
                          platform::errors::InvalidArgument(
                              "The %d-th Input(Param) have no elements."));

        void *g_data = nullptr;
        if (g->IsInitialized()) {
          PADDLE_ENFORCE_EQ(g->type(), dtype,
                            platform::errors::InvalidArgument(
                                "The %d-th Input(Param) and Input(Grad) should "
                                "have the same data type %s.",
                                i, framework::DataTypeToString(dtype)));
          PADDLE_ENFORCE_EQ(g->dims(), p->dims(),
                            platform::errors::InvalidArgument(
                                "The %d-th Input(Param) and Input(Grad) should "
                                "have the same shape.",
                                i));
          g_data = g_out->data();
        }

        ParamGradInfo *info;
        if (dtype == framework::proto::VarType::FP32) {
          fp32_infos.emplace_back();
          info = &fp32_infos.back();
        } else if (dtype == framework::proto::VarType::FP16) {
          fp16_infos.emplace_back();
          info = &fp16_infos.back();
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Unsupported data type %s.", framework::DataTypeToString(dtype)));
        }

        VLOG(10) << "Found " << framework::DataTypeToString(dtype)
                 << " parameter " << i << " shape=[" << p_out->dims()
                 << "] numel=" << numel << " grad.IsInitialized()="
                 << (g_out->IsInitialized() ? "true" : "false");

        info->param_t = p_out;
        info->grad_t = g_out;
        info->idx = i;
        info->numel = numel;
        info->numel_with_padding = 0;  // not determined yet
        info->numel_offset = 0;        // not determined yet
      }
    }
    VLOG(10) << "Fill ParamGradInfo ends";

    // Step 2: determine the numel_with_padding and numel_offset
    auto rank = ctx.Attr<int>("rank");
    auto nranks = ctx.Attr<int>("nranks");
    auto alignment = ctx.Attr<int>("alignment");
    VLOG(10) << "rank = " << rank << ", nranks = " << nranks
             << " , alignment = " << alignment;
    if (alignment <= 0) {
      alignment = platform::GpuMinChunkSize();
    }
    PADDLE_ENFORCE_GE(alignment, 1,
                      platform::errors::InvalidArgument(
                          "The attr(alignment) should be larger than 0."));
    PADDLE_ENFORCE_EQ(alignment & (alignment - 1), 0,
                      platform::errors::InvalidArgument(
                          "The attr(alignment) should be the power of 2."));
    PADDLE_ENFORCE_GE(
        rank, 0, platform::errors::InvalidArgument(
                     "The attr(rank) should be equal to or larger than 0."));
    PADDLE_ENFORCE_LT(
        rank, nranks,
        platform::errors::InvalidArgument(
            "The attr(rank) should be less than the attr(nranks)."));
    // NOTE: We guarantee that both fp32_numel and fp16_numel can be exactly
    // divided by alignment and nranks.
    auto fp32_numel = FillAlignmentPaddingInfo(&fp32_infos, alignment, nranks,
                                               framework::proto::VarType::FP32);
    VLOG(10) << "FP32 ParamGradInfo: " << string::join_strings(fp32_infos, " ");
    auto fp16_numel = FillAlignmentPaddingInfo(&fp16_infos, alignment, nranks,
                                               framework::proto::VarType::FP16);
    VLOG(10) << "FP16 ParamGradInfo: " << string::join_strings(fp16_infos, " ");
    auto total_numel = fp32_numel + fp16_numel;
    PADDLE_ENFORCE_LT(
        total_numel, std::numeric_limits<int>::max(),
        platform::errors::InvalidArgument("Too many parameter number."));

    auto fp32_numel_each_device = fp32_numel / nranks;
    auto fp16_numel_each_device = fp16_numel / nranks;
    auto numel_each_device = fp32_numel_each_device + fp16_numel_each_device;
    VLOG(10) << "Fill padding ends. total_numel = " << total_numel
             << ", fp32_numel = " << fp32_numel
             << ", fp16_numel = " << fp16_numel
             << ", fp32_numel_each_device = " << fp32_numel_each_device
             << ", fp16_numel_each_device = " << fp16_numel_each_device;

    // Step 3: allocate output tensor and do initialization
    float *fused_fp32_param = nullptr, *fused_fp32_grad = nullptr;
    platform::float16 *fused_fp16_param = nullptr, *fused_fp16_grad = nullptr;
    framework::Tensor *fp32_p_t = nullptr, *fp16_p_t = nullptr,
                      *fp32_g_t = nullptr, *fp16_g_t = nullptr;
    std::vector<framework::Tensor *> fp16_master_params;
    if (total_numel > 0) {
      fp32_p_t = ctx.Output<framework::Tensor>("FP32FusedParam");
      fused_fp32_param = TensorFillConstant<float>(
          dev_ctx, fp32_p_t, {static_cast<int64_t>(total_numel)}, 0.0f);
    }

    if (fp32_numel > 0) {
      fp32_g_t = ctx.Output<framework::Tensor>("FP32FusedGrad");
      fused_fp32_grad = TensorFillConstant<float>(
          dev_ctx, fp32_g_t, {static_cast<int64_t>(fp32_numel)}, 0.0f);
    }

    if (fp16_numel > 0) {
      fp16_p_t = ctx.Output<framework::Tensor>("FP16FusedParam");
      fused_fp16_param = TensorFillConstant<platform::float16>(
          dev_ctx, fp16_p_t, {static_cast<int64_t>(fp16_numel)},
          static_cast<platform::float16>(0));

      fp16_g_t = ctx.Output<framework::Tensor>("FP16FusedGrad");
      fused_fp16_grad = TensorFillConstant<platform::float16>(
          dev_ctx, fp16_g_t, {static_cast<int64_t>(fp16_numel)},
          static_cast<platform::float16>(0));
    }
    VLOG(10) << "Allocate FP32FusedParam/Grad, FP16FusedParam/Grad ends";

    // (1) For FP32FusedParam, memcpy for fp32 param and then share data, cast
    // for fp16 master weight
    // (2) For FP16FusedParam, memcpy and then share data
    // (3) For FP32FusedGrad/FP16FusedGrad, memcpy if gradient has been inited
    for (const auto &info : fp32_infos) {
      auto sliced_tensor = CopyAndShareBufferForInitedTensor(
          info.param_t, fp32_p_t, info.numel_offset, stream);
      master_params[info.idx]->Resize(info.param_t->dims());
      master_params[info.idx]->ShareBufferWith(sliced_tensor);
      PADDLE_ENFORCE_EQ(master_params[info.idx]->mutable_data<float>(),
                        sliced_tensor.data<float>());
      if (info.grad_t->IsInitialized()) {
        CopyAndShareBufferForInitedTensor(info.grad_t, fp32_g_t,
                                          info.numel_offset, stream);
      } else {
        ShareBufferForNonInitedTensor(info.grad_t, fp32_g_t, info.numel_offset,
                                      info.param_t->dims());
      }
    }

    size_t fp16_numel_offset = 0;
    if (fp32_numel > 0) {
      auto last_fp32_info = fp32_infos.back();
      fp16_numel_offset =
          last_fp32_info.numel_offset + last_fp32_info.numel_with_padding;
    }

    for (const auto &info : fp16_infos) {
      auto master_weight_offset = info.numel_offset + fp16_numel_offset;
      auto sliced_tensor = CastDataForInitedTensor(
          dev_ctx, info.param_t, fp32_p_t, master_weight_offset);
      master_params[info.idx]->Resize(info.param_t->dims());
      master_params[info.idx]->ShareBufferWith(sliced_tensor);

      CopyAndShareBufferForInitedTensor(info.param_t, fp16_p_t,
                                        info.numel_offset, stream);
      PADDLE_ENFORCE_EQ(master_params[info.idx]->mutable_data<float>(),
                        sliced_tensor.data<float>());

      if (info.grad_t->IsInitialized()) {
        CopyAndShareBufferForInitedTensor(info.grad_t, fp16_g_t,
                                          info.numel_offset, stream);
      } else {
        ShareBufferForNonInitedTensor(info.grad_t, fp16_g_t, info.numel_offset,
                                      info.param_t->dims());
      }
    }
    VLOG(10) << "Copy/share data for Param/Grad ends";

    // For Moment1, Moment2, Beta1Pow, Beta2Pow, just fill constant
    TensorFillConstant<float>(dev_ctx, ctx.Output<framework::Tensor>("Moment1"),
                              {static_cast<int64_t>(numel_each_device)}, 0.0f);
    TensorFillConstant<float>(dev_ctx, ctx.Output<framework::Tensor>("Moment2"),
                              {static_cast<int64_t>(numel_each_device)}, 0.0f);
    TensorFillConstant<float>(dev_ctx,
                              ctx.Output<framework::Tensor>("Beta1Pow"), {1},
                              ctx.Attr<float>("beta1"));
    TensorFillConstant<float>(dev_ctx,
                              ctx.Output<framework::Tensor>("Beta2Pow"), {1},
                              ctx.Attr<float>("beta2"));
    VLOG(10) << "Init Moment and BetaPow ends";

    // Do sharding
    size_t fp32_start_idx, fp32_end_idx, fp32_start_numel_offset,
        fp32_end_numel_offset;
    GetParamGradShardInfo(fp32_infos, rank * fp32_numel_each_device,
                          (rank + 1) * fp32_numel_each_device, &fp32_start_idx,
                          &fp32_end_idx, &fp32_start_numel_offset,
                          &fp32_end_numel_offset);
    size_t fp16_start_idx, fp16_end_idx, fp16_start_numel_offset,
        fp16_end_numel_offset;
    GetParamGradShardInfo(fp16_infos, rank * fp16_numel_each_device,
                          (rank + 1) * fp16_numel_each_device, &fp16_start_idx,
                          &fp16_end_idx, &fp16_start_numel_offset,
                          &fp16_end_numel_offset);
    size_t fp32_local_param_num =
        fp32_numel_each_device > 0 ? fp32_end_idx - fp32_start_idx + 1 : 0;
    size_t fp16_local_param_num =
        fp16_numel_each_device > 0 ? fp16_end_idx - fp16_start_idx + 1 : 0;
    size_t total_local_param_num = fp32_local_param_num + fp16_local_param_num;
    VLOG(10) << "Found the sharding arguments";

    // For LocalParamNum, perform H2H copy, but we
    // should find out the local
    // param number first!
    auto *local_param_info_t = ctx.Output<framework::Tensor>("LocalParamInfo");
    local_param_info_t->Resize({4});
    auto *local_param_info =
        local_param_info_t->mutable_data<int>(platform::CPUPlace());
    local_param_info[0] = static_cast<int>(fp32_start_idx);
    local_param_info[1] = static_cast<int>(fp32_local_param_num);
    local_param_info[2] = static_cast<int>(fp16_start_idx + fp32_infos.size());
    local_param_info[3] = static_cast<int>(fp16_local_param_num);
    VLOG(10) << "Local FP32 param: " << fp32_local_param_num
             << ", local FP16 param: " << fp16_local_param_num;

    // For WeightDecay, shard and perform H2D copy
    const auto &origin_weight_decay =
        ctx.Attr<std::vector<float>>("weight_decay");
    PADDLE_ENFORCE_EQ(params.size(), origin_weight_decay.size(),
                      platform::errors::InvalidArgument(
                          "The attr(weight_decay) should have the "
                          "same length "
                          "with Input(Param)."));
    std::vector<float> shard_weight_decay;
    shard_weight_decay.reserve(total_local_param_num);
    for (size_t i = 0; i < fp32_local_param_num; ++i) {
      shard_weight_decay.push_back(
          origin_weight_decay[fp32_infos[i + fp32_start_idx].idx]);
    }
    for (size_t i = 0; i < fp16_local_param_num; ++i) {
      shard_weight_decay.push_back(
          origin_weight_decay[fp16_infos[i + fp16_start_idx].idx]);
    }

    // For FusedIndices, launch CUDA kernel to do
    // binary search
    auto *fused_indices_t = ctx.Output<framework::Tensor>("FusedIndices");
    fused_indices_t->Resize({static_cast<int64_t>(total_numel)});
    auto *fused_indices = fused_indices_t->mutable_data<int>(place);
    std::vector<int> numel_offsets;
    numel_offsets.reserve(params.size() + 1);
    for (const auto &info : fp32_infos) {
      numel_offsets.push_back(info.numel_offset);
    }
    for (const auto &info : fp16_infos) {
      numel_offsets.push_back(info.numel_offset + fp16_numel_offset);
    }
    numel_offsets.push_back(fp32_numel + fp16_numel);
    PADDLE_ENFORCE_EQ(numel_offsets.size(), params.size() + 1,
                      platform::errors::InvalidArgument(
                          "The numel_offsets number must be one larger than "
                          "the parameter number."));
    VLOG(10) << "Total numel offset: " << FlattenToString(numel_offsets);
    auto *fused_param_offset_t =
        ctx.Output<framework::Tensor>("FusedParamOffsets");
    fused_param_offset_t->Resize({static_cast<int64_t>(numel_offsets.size())});
    auto *fused_param_offset = fused_param_offset_t->mutable_data<int>(place);
    memory::Copy(place, fused_param_offset, platform::CPUPlace(),
                 numel_offsets.data(),
                 numel_offsets.size() * sizeof(numel_offsets[0]), stream);
    auto config = platform::GetGpuLaunchConfig1D(dev_ctx, total_numel);
    LambFillFusedIndicesCUDAKernel<<<config.block_per_grid,
                                     config.thread_per_block, 0, stream>>>(
        fused_param_offset, fused_indices, numel_offsets.size() - 1,
        total_numel);

    std::vector<int> partial_numel_offsets;
    partial_numel_offsets.push_back(0);
    // Fill all padding indices
    std::vector<int> padding_indices_cpu;
    for (size_t i = fp32_start_idx; i < fp32_start_idx + fp32_local_param_num;
         ++i) {
      size_t start_n = fp32_infos[i].numel;
      size_t valid_start_n = 0;
      if (i == fp32_start_idx) {
        start_n = std::max(start_n, fp32_start_numel_offset);
        valid_start_n = fp32_start_numel_offset;
      }

      size_t end_n = fp32_infos[i].numel_with_padding;
      if (i + 1 == fp32_start_idx + fp32_local_param_num) {
        end_n = std::min(end_n, fp32_end_numel_offset);
      }

      auto offset = fp32_infos[i].numel_offset;
      for (size_t j = start_n; j < end_n; ++j) {
        padding_indices_cpu.push_back(offset + j);
      }

      PADDLE_ENFORCE_NE(valid_start_n, end_n);
      VLOG(10) << "FP32 Partial numel = ["
               << valid_start_n + fp32_infos[i].numel << ","
               << end_n + fp32_infos[i].numel;
      partial_numel_offsets.push_back(partial_numel_offsets.back() + end_n -
                                      valid_start_n);
    }

    for (size_t i = fp16_start_idx; i < fp16_start_idx + fp16_local_param_num;
         ++i) {
      size_t start_n = fp16_infos[i].numel;
      size_t valid_start_n = 0;
      if (i == fp16_start_idx) {
        start_n = std::max(start_n, fp16_start_numel_offset);
        valid_start_n = fp16_start_numel_offset;
      }

      size_t end_n = fp16_infos[i].numel_with_padding;
      if (i + 1 == fp16_start_idx + fp16_local_param_num) {
        end_n = std::min(end_n, fp16_end_numel_offset);
      }

      auto offset = fp16_numel_offset + fp16_infos[i].numel_offset;
      for (size_t j = start_n; j < end_n; ++j) {
        padding_indices_cpu.push_back(offset + j);
      }

      PADDLE_ENFORCE_NE(valid_start_n, end_n);
      VLOG(10) << "FP16 Partial numel = [" << valid_start_n + offset << ","
               << end_n + offset;
      partial_numel_offsets.push_back(partial_numel_offsets.back() + end_n -
                                      valid_start_n);
    }

    auto *partial_param_offset_t =
        ctx.Output<framework::Tensor>("PartialFusedParamOffsets");
    partial_param_offset_t->Resize(
        {static_cast<int64_t>(partial_numel_offsets.size())});
    auto *partial_param_offset =
        partial_param_offset_t->mutable_data<int>(place);
    memory::Copy(
        place, static_cast<void *>(partial_param_offset), platform::CPUPlace(),
        static_cast<const void *>(partial_numel_offsets.data()),
        partial_numel_offsets.size() * sizeof(partial_numel_offsets[0]),
        stream);

    std::vector<float> wd_cpu;
    for (size_t i = 0; i < shard_weight_decay.size(); ++i) {
      int len = partial_numel_offsets[i + 1] - partial_numel_offsets[i];
      for (int j = 0; j < len; ++j) {
        wd_cpu.push_back(shard_weight_decay[i]);
      }
    }
    PADDLE_ENFORCE_EQ(wd_cpu.size() * nranks, fp32_numel + fp16_numel);
    auto *weight_decay_t = ctx.Output<framework::Tensor>("WeightDecay");
    weight_decay_t->Resize({static_cast<int64_t>(wd_cpu.size())});
    auto *weight_decay = weight_decay_t->mutable_data<float>(place);
    memory::Copy(place, static_cast<void *>(weight_decay), platform::CPUPlace(),
                 static_cast<const void *>(wd_cpu.data()),
                 wd_cpu.size() * sizeof(wd_cpu[0]), stream);

    auto *global_scale = ctx.Output<framework::Tensor>("GlobalScale");
    if (!global_scale->IsInitialized()) {
      TensorFillConstant<float>(dev_ctx, global_scale, {1}, 1.0f);
    }
    VLOG(10) << "Init global scale ends";
    dev_ctx.Wait();
    VLOG(10) << "Wait for H2D copy";
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_CUDA_KERNEL(
    distributed_fused_lamb_init,
    ops::DistributedFusedLambInitOpKernel<plat::CUDADeviceContext, float>);
