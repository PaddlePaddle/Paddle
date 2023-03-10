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

#include <cmath>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/buffer.h"
#include "paddle/fluid/operators/amp/fp16_type_traits.h"
#include "paddle/fluid/operators/optimizers/distributed_fused_lamb_op.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/kernels/funcs/tensor_to_string.h"

namespace paddle {
namespace operators {

template <typename T>
using MasterT = typename details::MPTypeTrait<T>::Type;

using phi::funcs::FlattenToString;
using phi::funcs::ToVector;

template <typename T>
static void MultiTensorL2Norm(const phi::XPUContext &dev_ctx,
                              const T *x,
                              const int *offsets,
                              int n,
                              float *y) {
  // 求和不开方，最后统一开方
  for (int i = 0; i < n; i++) {
    auto length = offsets[i + 1] - offsets[i];
    int r = xpu::square_reduce_sum<T>(
        dev_ctx.x_context(), x + offsets[i], y + i, length, false);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "square_reduce_sum");
  }
}

template <typename ParamT, typename MasterT>
static void MultiTensorUpdateLambParamAndBetaPows(
    const phi::XPUContext &dev_ctx,
    const int *offsets,
    int n,
    const MasterT *trust_ratio_div,
    float lr,
    const MasterT *param_square_norm,
    const MasterT *trust_ratio_div_square_norm,
    ParamT *param,
    MasterT *master_param,
    float *beta1pow,
    float *beta2pow,
    float beta1,
    float beta2,
    bool kHasMasterParam) {
//   using xpu_fp16 = typename XPUTypeTrait<phi::dtype::float16>::Type;
  //   constexpr bool kHasMasterParam =
  //         !(std::is_same<ParamT, MasterT>::value);
  bool has_beta_pow = (beta1pow != nullptr);
  if (has_beta_pow) {
    PADDLE_ENFORCE_NOT_NULL(
        beta2pow,
        platform::errors::InvalidArgument("Beta2Pow should not be nullptr."));
  } else {
    PADDLE_ENFORCE_EQ(
        beta2pow,
        nullptr,
        platform::errors::InvalidArgument("Beta2Pow should be nullptr."));
  }

  if (kHasMasterParam) {
    PADDLE_ENFORCE_NOT_NULL(master_param,
                            platform::errors::InvalidArgument(
                                "master_param should not be nullptr."));

    for (int i = 0; i < n; i++) {
      auto length = offsets[i + 1] - offsets[i];
      int r = xpu::update_lamb_param_and_beta_pows<ParamT>(
          dev_ctx.x_context(),
          param + offsets[i],
          trust_ratio_div + offsets[i],
          param_square_norm + i,
          trust_ratio_div_square_norm + i,
          param + offsets[i],
          master_param + offsets[i],
          lr,
          length);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "update_lamb_param_and_beta_pows");
    }
  } else {
    for (int i = 0; i < n; i++) {
      auto length = offsets[i + 1] - offsets[i];
      int r = xpu::update_lamb_param_and_beta_pows<ParamT>(
          dev_ctx.x_context(),
          param + offsets[i],
          trust_ratio_div + offsets[i],
          param_square_norm + i,
          trust_ratio_div_square_norm + i,
          param + offsets[i],
          nullptr,
          lr,
          length);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "update_lamb_param_and_beta_pows");
    }
  }

  if (has_beta_pow) {
    // beta1pow_[0] *= beta1_;
    // beta2pow_[0] *= beta2_;
    // Todo: 变成cpu tensor
    int r = xpu::scale(dev_ctx.x_context(),
                       beta1pow,
                       beta1pow,
                       1,
                       false,  // bias_after_scale
                       beta1,  // scale,
                       0);     // bias
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    r = xpu::scale(dev_ctx.x_context(),
                   beta2pow,
                   beta2pow,
                   1,
                   false,  // bias_after_scale
                   beta2,  // scale,
                   0);     // bias
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
  }
}

template <int LogLevel>
static void LogParamAndTrustRatioDivSquareNorm(
    const framework::ExecutionContext &ctx,
    const float *param_square_norm,
    const float *trust_ratio_div_square_norm) {}

static bool IsFinite(const phi::XPUContext &dev_ctx, const float *ptr) {
  return false;
}

template <typename T>
static const T *GetInputTensorPtr(const framework::ExecutionContext &ctx,
                                  const char *in_name,
                                  int64_t *numel = nullptr) {
  const auto *in_tensor = ctx.Input<phi::DenseTensor>(in_name);
  PADDLE_ENFORCE_NOT_NULL(
      in_tensor,
      platform::errors::InvalidArgument("Input(%s) cannot be NULL.", in_name));
  if (in_tensor->IsInitialized()) {
    if (numel) *numel = in_tensor->numel();
    return in_tensor->data<T>();
  } else {
    if (numel) *numel = 0;
    return nullptr;
  }
}

template <typename T, bool AllowNotExist = false>
static T *GetSameInOutTensorPtr(const framework::ExecutionContext &ctx,
                                const platform::Place &place,
                                const char *in_name,
                                const char *out_name,
                                int64_t *numel = nullptr) {
  const auto *in_tensor = ctx.Input<phi::DenseTensor>(in_name);
  if (in_tensor == nullptr || !in_tensor->IsInitialized()) {
    PADDLE_ENFORCE_EQ(AllowNotExist,
                      true,
                      platform::errors::InvalidArgument(
                          "Input(%s) cannot be NULL.", in_name));
    if (numel) *numel = 0;
    return nullptr;
  }

  auto *out_tensor = ctx.Output<phi::DenseTensor>(out_name);
  PADDLE_ENFORCE_NOT_NULL(
      in_tensor,
      platform::errors::InvalidArgument("Input(%s) cannot be NULL.", in_name));
  PADDLE_ENFORCE_NOT_NULL(out_tensor,
                          platform::errors::InvalidArgument(
                              "Output(%s) cannot be NULL.", out_name));
  const T *in_data = in_tensor->data<T>();
  T *out_data = out_tensor->mutable_data<T>(place);
  PADDLE_ENFORCE_EQ(in_data,
                    out_data,
                    platform::errors::InvalidArgument(
                        "Input(%s) and Output(%s) must be the same Tensor.",
                        in_name,
                        out_name));
  if (numel) *numel = out_tensor->numel();
  return out_data;
}

template <typename T>
static bool HasNanInf(const phi::XPUContext &dev_ctx, const T *x, int numel) {
  return false;
}

static void CheckHasNanInfGrad(const float *fp32_grad,
                               int fp32_numel,
                               const float16 *fp16_grad,
                               int fp16_numel,
                               float *nan_inf_flag,
                               //    gpuStream_t stream,
                               memory::Buffer *cub_tmp_buffer) {}

void xpu2cpu(const phi::XPUContext &dev_ctx,
             const void *xpu_data,
             void *cpu_data,
             int len) {
  paddle::memory::Copy(phi::CPUPlace(),
                       cpu_data,
                       dev_ctx.GetPlace(),
                       xpu_data,
                       len);  // sizeof(T) * xxx
}

template <typename T>
class DistributedFusedLambOpKernel<phi::XPUContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    printf("====> run DistributedFusedLambOpKernel\n");

    using xpu_fp16 = typename XPUTypeTrait<phi::dtype::float16>::Type;

    auto &dev_ctx = ctx.template device_context<platform::XPUDeviceContext>();
    auto place = ctx.GetPlace();
    xpu::Context *xpu_ctx = dev_ctx.x_context();
    xpu::ctx_guard RAII_GUARD(xpu_ctx);

    auto *found_inf_t = ctx.Output<phi::DenseTensor>("FoundInf");
    found_inf_t->Resize({1});

    // Step 1: Get fp16 param and grad tensors
    int64_t fp16_numel;
    auto *fp16_param = reinterpret_cast<xpu_fp16 *>(
        GetSameInOutTensorPtr<platform::float16, true>(
            ctx, place, "FP16FusedParam", "FP16FusedParamOut", &fp16_numel));
    bool has_fp16_param = (fp16_numel > 0);
    const xpu_fp16 *fp16_grad = nullptr;
    if (has_fp16_param) {
      fp16_grad = reinterpret_cast<const xpu_fp16 *>(
          GetInputTensorPtr<platform::float16>(ctx, "FP16FusedGrad"));
    } else {
      fp16_param = nullptr;
    }

    // Step 2: Get fp32 param and grad tensors
    int64_t fp32_numel = 0;
    auto *fp32_param = GetSameInOutTensorPtr<float, true>(
        ctx, place, "FP32FusedParam", "FP32FusedParamOut", &fp32_numel);
    PADDLE_ENFORCE_GE(fp32_numel,
                      fp16_numel,
                      platform::errors::InvalidArgument(
                          "The element number in FP32FusedParam should be not "
                          "less than FP16FusedParam."));

    fp32_numel -= fp16_numel;  // the FP32FusedParam contains fp32 param and
                               // fp16 master weight
    bool has_fp32_param = (fp32_numel > 0);
    const float *fp32_grad = nullptr;
    if (has_fp32_param) {
      fp32_grad = GetInputTensorPtr<float>(ctx, "FP32FusedGrad");
    } else {
      PADDLE_ENFORCE_EQ(
          has_fp16_param,
          true,
          platform::errors::InvalidArgument(
              "Either FP32FusedGrad or FP16FusedGrad cannot be NULL."));
    }

    auto numel = fp32_numel + fp16_numel;
    VLOG(1) << "numel = " << numel << " , fp32_numel = " << fp32_numel
            << " , fp16_numel = " << fp16_numel;

    // The NVIDIA cub library does not support number > INT32_MAX
    PADDLE_ENFORCE_LE(numel,
                      std::numeric_limits<int>::max(),
                      platform::errors::Unimplemented(
                          "Too many parameter number. Only <= %d is supported.",
                          std::numeric_limits<int>::max()));

    auto acc_steps = ctx.Attr<int>("acc_steps");
    PADDLE_ENFORCE_GE(
        acc_steps,
        1,
        platform::errors::InvalidArgument(
            "The gradient accumulation steps should be not less than 1."));

    // 梯度累计大于 1
    if (acc_steps > 1) {
      assert(0);
      auto *step_t = ctx.Output<phi::DenseTensor>("AccStep");
      PADDLE_ENFORCE_NOT_NULL(
          step_t,
          platform::errors::InvalidArgument(
              "Output(AccStep) cannot be nullptr when Attr(acc_steps) > 1."));
      bool is_initialized = step_t->IsInitialized();
      int64_t *step_ptr;
      if (is_initialized) {
        step_ptr = step_t->mutable_data<int64_t>(platform::CPUPlace());
        ++(*step_ptr);
      } else {
        step_t->Resize({1});
        step_ptr = step_t->mutable_data<int64_t>(platform::CPUPlace());
        *step_ptr = 1;
      }
      // 梯度累计，循环计数, acc_steps = 1
      int64_t rounded_step = (*step_ptr) % acc_steps;

      float *fp32_acc_grad = nullptr;
      if (has_fp32_param) {
        auto *fp32_acc_grad_t =
            ctx.Output<phi::DenseTensor>("FP32AccFusedGrad");
        PADDLE_ENFORCE_NOT_NULL(
            fp32_acc_grad_t,
            platform::errors::InvalidArgument(
                "Output(FP32AccFusedGrad) cannot be nullptr "
                "when Attr(acc_steps) > 1."));
        if (!fp32_acc_grad_t->IsInitialized()) {
          fp32_acc_grad_t->Resize({static_cast<int64_t>(fp32_numel)});
          fp32_acc_grad = fp32_acc_grad_t->mutable_data<float>(place);
        } else {
          fp32_acc_grad = fp32_acc_grad_t->data<float>();
        }
      }

      xpu_fp16 *fp16_acc_grad = nullptr;
      float *master_acc_grad = nullptr;
      (void)master_acc_grad;
      bool use_master_acc_grad = false;
      if (has_fp16_param) {
        // True, 梯度累积用 float类型
        use_master_acc_grad = ctx.Attr<bool>("use_master_acc_grad");
        auto *fp16_acc_grad_t =
            ctx.Output<phi::DenseTensor>("FP16AccFusedGrad");
        PADDLE_ENFORCE_NOT_NULL(
            fp16_acc_grad_t,
            platform::errors::InvalidArgument(
                "Output(FP16AccFusedGrad) cannot be nullptr "
                "when Attr(acc_steps) > 1."));
        if (!fp16_acc_grad_t->IsInitialized()) {
          /* use_master_acc_grad = true, fp16 grad acc 备份一份 fp32数据 */
          auto acc_grad_size =
              use_master_acc_grad ? (3 * fp16_numel) : fp16_numel;
          fp16_acc_grad_t->Resize({static_cast<int64_t>(acc_grad_size)});
          fp16_acc_grad = reinterpret_cast<xpu_fp16 *>(
              fp16_acc_grad_t->mutable_data<platform::float16>(place));
        } else {
          fp16_acc_grad = reinterpret_cast<xpu_fp16 *>(
              fp16_acc_grad_t->data<platform::float16>());
        }
        if (use_master_acc_grad) {
          master_acc_grad =
              reinterpret_cast<float *>(fp16_acc_grad + fp16_numel);
        }
      }

      // Inplace addto
      if (has_fp32_param) {
        if (rounded_step == 1) {
        //   memory::Copy(place,
        //                fp32_acc_grad,
        //                place,
        //                fp32_grad,
        //                fp32_numel * sizeof(float));
            int r = xpu::copy(xpu_ctx, fp32_grad, fp32_acc_grad, fp32_numel);
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
        } else {
          //   LaunchElementwiseAddWithCastKernel(dev_ctx,
          //                                      fp32_grad,
          //                                      fp32_acc_grad,
          //                                      fp32_acc_grad,
          //                                      fp32_numel,
          //                                      stream);
          // Todo:
          int r = xpu::elementwiseadd_with_cast(xpu_ctx, fp32_grad, fp32_acc_grad, fp32_acc_grad, fp32_numel);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "elementwiseadd_with_cast");
        }
      }

      if (has_fp16_param) {
        if (acc_steps == 2 || !use_master_acc_grad) {
          if (rounded_step != 1) {
            // LaunchElementwiseAddWithCastKernel(dev_ctx,
            //                                    fp16_acc_grad,
            //                                    fp16_grad,
            //                                    fp16_acc_grad,
            //                                    fp16_numel,
            //                                    stream);
            // Todo:
            int r = xpu::elementwiseadd_with_cast<xpu_fp16, xpu_fp16, xpu_fp16>(xpu_ctx, fp16_grad, fp16_acc_grad, fp16_acc_grad, fp16_numel);
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "elementwiseadd_with_cast");
          } else {
            // memory::Copy(place,
            //              fp16_acc_grad,
            //              place,
            //              fp16_grad,
            //              fp16_numel * sizeof(float16));
            int r = xpu::copy(xpu_ctx, fp16_grad, fp16_acc_grad, fp16_numel);
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
          }
        } else {  // acc_steps >= 3
        //   float *fp32_acc_grad_tmp = RAII_GUARD.alloc<float>(fp16_numel);
          if (rounded_step == 0) {
            // fp32 转化为 fp16 更新
            // fp16_acc_grad = master_acc_grad + fp16_grad （fp16 = float + fp16)
            // LaunchElementwiseAddWithCastKernel(dev_ctx,
            //                                    fp16_grad,
            //                                    master_acc_grad,
            //                                    fp16_acc_grad,
            //                                    fp16_numel,
            //                                    stream);
            // Todo:
            int r = xpu::elementwiseadd_with_cast<float, xpu_fp16, xpu_fp16>(xpu_ctx, master_acc_grad, fp16_grad, fp16_acc_grad, fp16_numel);
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "elementwiseadd_with_cast");
          } else if (rounded_step == 1) {
            // copy: fp16_grad ===> fp16_acc_grad (fp16 ==> fp16)
            // memory::Copy(place,
            //              fp16_acc_grad,
            //              place,
            //              fp16_grad,
            //              fp16_numel * sizeof(float16));
            int r = xpu::copy(xpu_ctx, fp16_grad, fp16_acc_grad, fp16_numel);
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "copy");
          } else if (rounded_step == 2) {
            // master_acc_grad = fp16_acc_grad + fp16_grad （float = fp16 + fp16)
            // LaunchElementwiseAddWithCastKernel(dev_ctx,
            //                                    fp16_grad,
            //                                    fp16_acc_grad,
            //                                    master_acc_grad,
            //                                    fp16_numel,
            //                                    stream);
            // Todo:
            int r = xpu::elementwiseadd_with_cast<xpu_fp16, xpu_fp16, float>(xpu_ctx, fp16_acc_grad, fp16_grad, master_acc_grad, fp16_numel);
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "elementwiseadd_with_cast");
          } else {
            // master_acc_grad = master_acc_grad + fp16_grad (float = float + fp16)
            // LaunchElementwiseAddWithCastKernel(dev_ctx,
            //                                    fp16_grad,
            //                                    master_acc_grad,
            //                                    master_acc_grad,
            //                                    fp16_numel,
            //                                    stream);
            // Todo:
            int r = xpu::elementwiseadd_with_cast<float, xpu_fp16, float>(xpu_ctx, master_acc_grad, fp16_grad, master_acc_grad, fp16_numel);
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "elementwiseadd_with_cast");
          }
        }
      }

      auto *stop_update_t = ctx.Output<phi::DenseTensor>("StopUpdate");
      stop_update_t->Resize({1});
      auto *stop_update =
          stop_update_t->mutable_data<bool>(platform::CPUPlace());

      auto *found_inf_cpu =
          found_inf_t->mutable_data<bool>(platform::CPUPlace());

      if (rounded_step != 0) {
        *stop_update = true;
        auto *found_inf_cpu =
            found_inf_t->mutable_data<bool>(platform::CPUPlace());
        *found_inf_cpu = false;
        return;
      } else {
        // swap pointer
        fp32_grad = fp32_acc_grad;
        fp16_grad = fp16_acc_grad;
        *stop_update = false;
        found_inf_t->clear();
      }
      (void)found_inf_cpu;
    }
    // 梯度累计结束

    // Step 3: Get ParamInfo
    const auto *param_info_tensor = GetInputTensorPtr<int>(ctx, "ParamInfo");
    auto fp32_local_start_idx = param_info_tensor[0];
    auto fp32_local_param_num = param_info_tensor[1];
    auto fp32_global_param_num = param_info_tensor[2];
    auto fp32_weight_decay_end_idx = param_info_tensor[3];
    auto fp16_local_start_idx = param_info_tensor[4];
    auto fp16_local_param_num = param_info_tensor[5];
    auto fp16_global_param_num = param_info_tensor[6];
    auto fp16_weight_decay_end_idx = param_info_tensor[7];

    auto local_param_num = fp32_local_param_num + fp16_local_param_num;
    auto param_num = fp32_global_param_num + fp16_global_param_num;
    PADDLE_ENFORCE_LE(local_param_num,
                      param_num,
                      platform::errors::InvalidArgument(
                          "The local parameter number should not exceed the "
                          "global parameter number."));
    VLOG(1) << "local_param_num = " << local_param_num
            << " , global_param_num = " << param_num
            << " , fp32_local_start_idx = " << fp32_local_start_idx
            << " , fp32_local_param_num = " << fp32_local_param_num
            << " , fp32_global_param_num = " << fp32_global_param_num
            << " , fp16_local_start_idx = " << fp16_local_start_idx
            << " , fp16_local_param_num = " << fp16_local_param_num
            << " , fp16_global_param_num = " << fp16_global_param_num;

    // Step 4: Get LearningRate, Moment1, Moment2, Beta1Pow, Beta2Pow,
    // GlobalScale
    const auto *global_scale = GetInputTensorPtr<float>(ctx, "GlobalScale");
    const auto *lr = GetInputTensorPtr<float>(ctx, "LearningRate");
    int64_t partial_numel = 0;
    auto *moment1 = GetSameInOutTensorPtr<float>(
        ctx, place, "Moment1", "Moment1Out", &partial_numel);

    // loss scalling value
    float cpu_global_scale, cpu_lr;
    xpu2cpu(dev_ctx, global_scale, &cpu_global_scale, sizeof(float));
    printf("===> cpu_global_scale: %f\n", cpu_global_scale);
    xpu2cpu(dev_ctx, lr, &cpu_lr, sizeof(float));
    printf("===> cpu_lr: %f\n", cpu_lr);

    PADDLE_ENFORCE_EQ(numel % partial_numel,
                      0,
                      platform::errors::InvalidArgument(
                          "The total parameter number %d should be divided "
                          "exactly by the element number %d of Moment1.",
                          numel,
                          partial_numel));

    // The num_devices means the number of devices that shard a complete set
    // of all parameters. It may be num_devices < nranks or num_devices ==
    // nranks.
    int64_t num_devices = numel / partial_numel;
    VLOG(1) << "num_devices = " << num_devices
            << " , partial_numel = " << partial_numel;

    std::cout << "num_devices = " << num_devices
              << " , partial_numel = " << partial_numel << std::endl;

    PADDLE_ENFORCE_EQ(fp32_numel % num_devices,
                      0,
                      platform::errors::InvalidArgument(
                          "The fp32 parameter number %d should be divided "
                          "exactly by the device number %d.",
                          fp32_numel,
                          num_devices));
    PADDLE_ENFORCE_EQ(fp16_numel % num_devices,
                      0,
                      platform::errors::InvalidArgument(
                          "The fp16 parameter number %d should be divided "
                          "exactly by the device number %d.",
                          fp16_numel,
                          num_devices));

    auto *moment2 =
        GetSameInOutTensorPtr<float>(ctx, place, "Moment2", "Moment2Out");
    auto *beta1pow =
        GetSameInOutTensorPtr<float>(ctx, place, "Beta1Pow", "Beta1PowOut");
    auto *beta2pow =
        GetSameInOutTensorPtr<float>(ctx, place, "Beta2Pow", "Beta2PowOut");

    auto *found_inf = found_inf_t->mutable_data<bool>(place);

    // Step 5: Get attributes weight_decay, beta1, beta2, epsilon,
    // max_grad_norm, ring_id,
    // use_master_param_norm, is_grad_scaled_by_nranks
    auto weight_decay = ctx.Attr<float>("weight_decay");
    auto beta1 = ctx.Attr<float>("beta1");
    auto beta2 = ctx.Attr<float>("beta2");
    auto epsilon = ctx.Attr<float>("epsilon");
    // args.max_grad_norm: 默认1
    auto max_global_grad_norm = ctx.Attr<float>("max_global_grad_norm");
    // python，配置为 False
    auto clip_after_allreduce = ctx.Attr<bool>("clip_after_allreduce");
    auto nranks = ctx.Attr<int64_t>("nranks");
    PADDLE_ENFORCE_GE(nranks,
                      num_devices,
                      phi::errors::InvalidArgument(
                          "The nranks must be not less than num_devices."));
    PADDLE_ENFORCE_EQ(
        nranks % num_devices,
        0,
        phi::errors::InvalidArgument(
            "The nranks must be exactly divided by num_devices."));
    bool local_shard = (nranks > num_devices);
    // Todo:
    PADDLE_ENFORCE_EQ(local_shard,
                      false,
                      phi::errors::InvalidArgument("local_shard must False."));

    const auto &ring_ids = ctx.Attr<std::vector<int>>("ring_id");
    (void)ring_ids;
    // // python，配置为 True
    auto use_master_param_norm = ctx.Attr<bool>("use_master_param_norm");
    auto is_grad_scaled_by_nranks = ctx.Attr<bool>("is_grad_scaled_by_nranks");
    auto use_hierarchical_allreduce =
        ctx.Attr<bool>("use_hierarchical_allreduce");
    VLOG(10) << "max_global_grad_norm = " << max_global_grad_norm
             << " , clip_after_allreduce = " << clip_after_allreduce
             << " , use_master_param_norm = " << use_master_param_norm
             << " , is_grad_scaled_by_nranks = " << is_grad_scaled_by_nranks
             << " , local_shard = " << local_shard
             << " , use_hierarchical_allreduce = "
             << use_hierarchical_allreduce;

    // Step 6: allreduce + global norm gradient clip
    // int64_t global_rank = 0, local_rank = 0;
    // ncclComm_t global_comm = nullptr, local_comm = nullptr,
    //            external_comm = nullptr;
    // if (nranks > 1) {
    //   auto *nccl_comm_handle =
    //       platform::NCCLCommContext::Instance().Get(ring_ids[0], place);
    //   global_comm = nccl_comm_handle->comm();
    //   global_rank = nccl_comm_handle->rank();

    //   if (local_shard) {
    //     auto *local_nccl_comm_handle =
    //         platform::NCCLCommContext::Instance().Get(ring_ids[1], place);
    //     local_comm = local_nccl_comm_handle->comm();
    //     local_rank = local_nccl_comm_handle->rank();
    //     if (use_hierarchical_allreduce) {
    //       external_comm = platform::NCCLCommContext::Instance()
    //                           .Get(ring_ids[2], place)
    //                           ->comm();
    //     }
    //   } else {
    //     local_comm = global_comm;
    //     local_rank = global_rank;
    //   }
    // }

    memory::Buffer grad_norm_square_buffer(place);
    auto *fp32_square_grad_norm = grad_norm_square_buffer.Alloc<float>(2);
    memory::Buffer cub_tmp_buffer(place);

    memory::Buffer sum_grad_buffer(place);
    // norm clip 后的 grad
    float *fp32_sum_grad;
    (void)fp32_sum_grad;
    xpu_fp16 *fp16_sum_grad;
    (void)fp16_sum_grad;
    auto fp32_numel_each_device = fp32_numel / num_devices;
    auto fp16_numel_each_device = fp16_numel / num_devices;
    if (local_shard) {
      auto ptr = sum_grad_buffer.Alloc<uint8_t>(fp32_numel * sizeof(float) +
                                                fp16_numel * sizeof(float16));
      fp32_sum_grad = has_fp32_param ? reinterpret_cast<float *>(ptr) : nullptr;
      fp16_sum_grad =
          has_fp16_param
              ? reinterpret_cast<xpu_fp16 *>(ptr + fp32_numel * sizeof(float))
              : nullptr;
    } else if (nranks > 1 ||
               (max_global_grad_norm > 0 && !clip_after_allreduce)) {
      auto ptr = sum_grad_buffer.Alloc<uint8_t>(
          fp32_numel_each_device * sizeof(float) +
          fp16_numel_each_device * sizeof(float16));
      fp32_sum_grad = has_fp32_param ? reinterpret_cast<float *>(ptr) : nullptr;
      fp16_sum_grad = has_fp16_param
                          ? reinterpret_cast<xpu_fp16 *>(
                                ptr + fp32_numel_each_device * sizeof(float))
                          : nullptr;
    } else {
      // NOTE: The const_cast here is not important. The fp32_sum_grad and
      // fp16_sum_grad would not be changed when num_devices == 1
      // But if I do not perform const_cast here, there would be more
      // if-else codes (num_devices > 1) when I write the following code.
      // So I prefer to use const_cast to unify the following code to reduce
      // the if-else codes.
      fp32_sum_grad = const_cast<float *>(fp32_grad);
      fp16_sum_grad = const_cast<xpu_fp16 *>(fp16_grad);
    }

    float rescale_grad = 1.0f;
    if (!is_grad_scaled_by_nranks) {
      rescale_grad /= nranks;
    }

    int fp32_nums_inf_nans = 0;
    int fp16_nums_inf_nans = 0;

    if (max_global_grad_norm > 0) {
      if (clip_after_allreduce) {
        assert(0);
      } else {
        // (1) Calculate the local grad norm
        // GetSquareGradNorm(fp32_grad,
        //                   fp32_numel,
        //                   fp16_grad,
        //                   fp16_numel,
        //                   fp32_square_grad_norm,
        //                   stream,
        //                   &cub_tmp_buffer);

        printf("fp32_numel: %ld, fp16_numel: %ld\n", fp32_numel, fp16_numel);

        if (fp32_numel > 0) {
          int r = xpu::square_reduce_sum<float>(
              xpu_ctx, fp32_grad, fp32_square_grad_norm, fp32_numel, false);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "square_reduce_sum");
        }

        if (fp16_numel > 0) {
          float *fp16_square_norm = fp32_numel > 0 ? fp32_square_grad_norm + 1
                                                   : fp32_square_grad_norm;
          int r = xpu::square_reduce_sum<xpu_fp16>(
              xpu_ctx, fp16_grad, fp16_square_norm, fp16_numel, false);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "square_reduce_sum");

          float cpu_fp16_square_norm = 0.0f;
          xpu2cpu(
              dev_ctx, fp16_square_norm, &cpu_fp16_square_norm, sizeof(float));
          printf("===> cpu_fp16_square_norm: %f\n", cpu_fp16_square_norm);

          if (fp32_numel > 0) {
            // add(Context* ctx, const T* x, const T* y, T* z, int64_t len)
            r = xpu::add<float>(xpu_ctx,
                                fp16_square_norm,
                                fp32_square_grad_norm,
                                fp32_square_grad_norm,
                                1);
            PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
          }
        }

        // VLOG(1) << "Grad square norm before all reduce: "
        //         << FlattenToString(fp32_square_grad_norm, 1, place);
        // (2) Calculate the gradient clip scale
        float *fp32_scale = nullptr;
        float16 *fp16_scale = nullptr;
        if (has_fp32_param && has_fp16_param) {
          auto *ptr =
              cub_tmp_buffer.Alloc<uint8_t>(sizeof(float) + sizeof(float16));
          fp32_scale = reinterpret_cast<float *>(ptr);
          fp16_scale = reinterpret_cast<float16 *>(ptr + sizeof(float));
        } else if (has_fp32_param) {
          fp32_scale = cub_tmp_buffer.Alloc<float>(1);
        } else {
          fp16_scale = cub_tmp_buffer.Alloc<float16>(1);
        }
        (void)fp32_scale;
        (void)fp16_scale;

        float clip_scale = 1.0f;
        // python，配置为 False
        if (is_grad_scaled_by_nranks) {
          clip_scale *= nranks;
        }
        // CalcGradNormClipBeforeAllReduceScale<float, float16>
        //     <<<1, 1, 0, stream>>>(global_scale,
        //                           max_global_grad_norm,
        //                           fp32_square_grad_norm,
        //                           fp32_scale,
        //                           fp16_scale,
        //                           clip_scale);

        // check_finite_and_unscale
        float cpu_fp32_square_grad_norm, cpu_fp32_scale;
        xpu2cpu(dev_ctx,
                fp32_square_grad_norm,
                &cpu_fp32_square_grad_norm,
                sizeof(float));

        float grad_norm = std::sqrt(cpu_fp32_square_grad_norm) * clip_scale;
        cpu_fp32_scale =
            cpu_global_scale * max_global_grad_norm / (1e-6 + grad_norm);
        bool found_nan_inf = !isfinite(cpu_fp32_scale);
        // grad_norm 很大的话, cpu_fp32_scale 较小
        if (cpu_fp32_scale >= 1 || found_nan_inf) {
          cpu_fp32_scale = 1.0f;
        }

        printf("===> fp32_scale: %f\n", cpu_fp32_scale);

        // (3) Do ReduceScatter with scale
        VLOG(1) << "FP32 HasNanInf before all reduce: "
                << HasNanInf(dev_ctx, fp32_grad, fp32_numel);
        VLOG(1) << "FP16 HasNanInf before all reduce: "
                << HasNanInf(dev_ctx, fp16_grad, fp16_numel);
        if (local_shard) {
          assert(0);
        } else {
          // fp32_sum_grad * fp32_scale
          //   NCCLReduceScatterWithScale(fp32_grad,
          //                              fp32_sum_grad,
          //                              fp32_numel_each_device,
          //                              nranks,
          //                              global_comm,
          //                              stream,
          //                              dev_ctx,
          //                              fp32_scale);
          //   NCCLReduceScatterWithScale(fp16_grad,
          //                              fp16_sum_grad,
          //                              fp16_numel_each_device,
          //                              nranks,
          //                              global_comm,
          //                              stream,
          //                              dev_ctx,
          //                              fp16_scale);
          int r = xpu::scale(xpu_ctx,
                             fp32_grad,
                             fp32_sum_grad,
                             fp32_numel_each_device,
                             false,           // bias_after_scale
                             cpu_fp32_scale,  // *fp32_scale,
                             0);              // bias
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
          r = xpu::scale(xpu_ctx,
                         fp16_grad,
                         fp16_sum_grad,
                         fp16_numel_each_device,
                         false,           // bias_after_scale
                         cpu_fp32_scale,  // *fp16_scale,
                         0);              // bias
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
        }
        VLOG(1) << "FP32 HasNanInf after all reduce: "
                << HasNanInf(dev_ctx, fp32_sum_grad, fp32_numel_each_device);
        VLOG(1) << "FP16 HasNanInf after all reduce: "
                << HasNanInf(dev_ctx, fp16_sum_grad, fp16_numel_each_device);
        // =======>
        // CheckHasNanInfGrad(fp32_sum_grad,
        //                    fp32_numel_each_device,
        //                    fp16_sum_grad,
        //                    fp16_numel_each_device,
        //                    fp32_square_grad_norm,
        //                    &cub_tmp_buffer);
        if (fp32_numel > 0) {
          int r =
              count_nan_or_inf(xpu_ctx,
                               fp32_sum_grad,
                               reinterpret_cast<int *>(fp32_square_grad_norm),
                               fp32_numel_each_device);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "count_nan_or_inf");
          paddle::memory::Copy(phi::CPUPlace(),
                               &fp32_nums_inf_nans,
                               dev_ctx.GetPlace(),
                               fp32_square_grad_norm,
                               sizeof(int));
          std::cout << "fp32_nums_inf_nans: " << fp32_nums_inf_nans
                    << std::endl;
        }
        if (fp16_numel > 0) {
          int r =
              count_nan_or_inf(xpu_ctx,
                               fp16_sum_grad,
                               reinterpret_cast<int *>(fp32_square_grad_norm),
                               fp16_numel_each_device);
          PADDLE_ENFORCE_XDNN_SUCCESS(r, "count_nan_or_inf");
          paddle::memory::Copy(phi::CPUPlace(),
                               &fp16_nums_inf_nans,
                               dev_ctx.GetPlace(),
                               fp32_square_grad_norm,
                               sizeof(int));
          std::cout << "fp16_nums_inf_nans: " << fp16_nums_inf_nans
                    << std::endl;
        }

        if (num_devices > 1) {
          assert(0);
        }
        // (4) mark max_global_grad_norm as 0, meaning that clip has been
        // already performed
        max_global_grad_norm = 0;
      }
    } else {
      assert(0);
      max_global_grad_norm = 0;
    }
    VLOG(10) << "ReduceScatter done";

    // Step 7: update the moment1, moment2. Calcuate the trust_ratio_div
    auto *fused_offsets_t = ctx.Input<phi::DenseTensor>("FusedParamOffsets");
    auto *fused_offsets = fused_offsets_t->data<int>();
    auto *fp32_partial_fused_offsets_t =
        ctx.Input<phi::DenseTensor>("FP32ShardFusedParamOffsets");
    const auto *fp32_partial_fused_offsets =
        fp32_partial_fused_offsets_t->data<int>();
    auto *fp16_partial_fused_offsets_t =
        ctx.Input<phi::DenseTensor>("FP16ShardFusedParamOffsets");
    const auto *fp16_partial_fused_offsets =
        fp16_partial_fused_offsets_t->data<int>();

    auto *step = ctx.Output<phi::DenseTensor>("Step")->data<int64_t>();

    (void)fused_offsets;
    // VLOG(1) << "FusedParamOffsets: "
    //         << FlattenToString(fused_offsets,
    //                            fused_offsets_t->numel(),
    //                            fused_offsets_t->place());
    // VLOG(1) << "FP32ShardFusedParamOffsets: "
    //         << FlattenToString(fp32_partial_fused_offsets,
    //                            fp32_partial_fused_offsets_t->numel(),
    //                            fp32_partial_fused_offsets_t->place());
    // VLOG(1) << "FP16ShardFusedParamOffsets: "
    //         << FlattenToString(fp16_partial_fused_offsets,
    //                            fp16_partial_fused_offsets_t->numel(),
    //                            fp16_partial_fused_offsets_t->place());

    int64_t local_rank = 0;  // fc add
    memory::Buffer trust_ratio_div_buffer(place);
    auto *trust_ratio_div = trust_ratio_div_buffer.Alloc<float>(partial_numel);
    auto fp32_offset = local_rank * fp32_numel_each_device;
    auto fp16_offset = local_rank * fp16_numel_each_device;

    float beta1pow_cpu, beta2pow_cpu;
    xpu2cpu(dev_ctx, beta1pow, &beta1pow_cpu, sizeof(float));
    xpu2cpu(dev_ctx, beta2pow, &beta2pow_cpu, sizeof(float));

    printf("===> beta1pow: %f, beta2pow: %f\n", beta1pow_cpu, beta2pow_cpu);

    // T square_grad_norm = *square_grad_norm_p;
    // bool need_update_found_inf =
    //     (found_inf && threadIdx.x == 0 && blockIdx.x == 0);
    // if (!isfinite(square_grad_norm)) {
    //     if (need_update_found_inf) *found_inf = true;
    //     return;
    // } else if (need_update_found_inf) {
    //     *found_inf = false;
    //     ++(*step);
    // }

    // T scale = rescale_grad / global_scale[0];
    // if (max_global_grad_norm > 0) {
    //     T clip_scale =
    //         max_global_grad_norm / (sqrtf(square_grad_norm) * scale + 1e-6);
    //     if (clip_scale < static_cast<T>(1)) {
    //     scale *= clip_scale;
    //     }
    // }

    // Todo: 哪里用？ step 改为 GPU 变量
    if (fp32_nums_inf_nans > 0 || fp16_nums_inf_nans > 0) {
      // *found_inf = true;
      int r = xpu::constant(xpu_ctx, found_inf, 1, true);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
    } else {
      // *found_inf = false;
      // ++(*step);
      int r = xpu::constant(xpu_ctx, found_inf, 1, false);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "constant");
      r = xpu::scale(xpu_ctx,
                     step,
                     step,
                     1,
                     false,  // bias_after_scale
                     1.0,    // *fp32_scale,
                     1.0);   // bias
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");

      int64_t cpu_step;
      xpu2cpu(dev_ctx, step, &cpu_step, sizeof(int64_t));
      printf("===> cpu_step: %ld\n", cpu_step);
    }

    float update_scale = rescale_grad / cpu_global_scale;
    // Todo: 以下逻辑在 MultiTensorUpdateLambMomentAndTrustRatioDiv
    // if (max_global_grad_norm > 0) {
    //   float clip_scale =
    //       // max_global_grad_norm / (sqrtf(square_grad_norm) * scale + 1e-6);
    //       max_global_grad_norm / (sqrtf(0.0f) * update_scale + 1e-6);
    //   if (clip_scale < static_cast<T>(1)) {
    //     update_scale *= clip_scale;
    //   }
    // }

    printf("==> update_scale: %f\n", update_scale);

    // fp16_sum_grad cast to fp32
    // float* fp16_sum_grad_tmp = nullptr;

    if (has_fp32_param && (fp32_nums_inf_nans == 0)) {
      //   VLOG(10) << "Update FP32 Moment and TrustRatioDiv starts";
      //   MultiTensorUpdateLambMomentAndTrustRatioDiv(dev_ctx,
      //                                               fp32_partial_fused_offsets,
      //                                               fp32_local_param_num,
      //                                               fp32_param + fp32_offset,
      //                                               fp32_sum_grad,
      //                                               fp32_square_grad_norm, //
      //                                               square_grad_norm_p
      //                                               global_scale,
      //                                               beta1pow,
      //                                               beta2pow,
      //                                               moment1,
      //                                               moment2,
      //                                               trust_ratio_div,
      //                                               found_inf,
      //                                               step,
      //                                               weight_decay,
      //                                               fp32_weight_decay_end_idx,
      //                                               beta1,
      //                                               beta2,
      //                                               epsilon,
      //                                               max_global_grad_norm,
      //                                               rescale_grad);
      //   VLOG(10) << "Update FP32 Moment and TrustRatioDiv done";

      int numel = fp32_partial_fused_offsets[fp32_local_param_num] -
                  fp32_partial_fused_offsets[0];
      int weight_decay_end_numel =
          fp32_partial_fused_offsets[fp32_weight_decay_end_idx] -
          fp32_partial_fused_offsets[0];
      printf(
          "fp32 num: %d, fp32_local_param_num: %d, weight_decay_end_numel: "
          "%d\n",
          numel,
          fp32_local_param_num,
          weight_decay_end_numel);

      int r = xpu::update_lamb_mom_and_trust_ratio_div<float, float>(
          xpu_ctx,
          fp32_param + fp32_offset,
          fp32_sum_grad,
          moment1,
          moment2,
          trust_ratio_div,
          1 - beta1pow_cpu,
          1 - beta2pow_cpu,
          weight_decay,
          weight_decay_end_numel,
          beta1,
          beta2,
          epsilon,
          update_scale,
          numel);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "update_lamb_mom_and_trust_ratio_div");
    }
    float *master_param = nullptr;
    master_param = fp32_param + fp32_numel;
    if (has_fp16_param && (fp16_nums_inf_nans == 0)) {
      //   master_param = fp32_param + fp32_numel;
      //   VLOG(10) << "Update FP16 Moment and TrustRatioDiv starts";
      //   auto tmp_found_inf = has_fp32_param ? nullptr : found_inf;
      //   auto tmp_step = has_fp32_param ? nullptr : step;
      //   MultiTensorUpdateLambMomentAndTrustRatioDiv(
      //       dev_ctx,
      //       fp16_partial_fused_offsets,
      //       fp16_local_param_num,
      //       master_param + fp16_offset,
      //       fp16_sum_grad,
      //       fp32_square_grad_norm,
      //       global_scale,
      //       beta1pow,
      //       beta2pow,
      //       moment1 + fp32_numel_each_device,
      //       moment2 + fp32_numel_each_device,
      //       trust_ratio_div + fp32_numel_each_device,
      //       tmp_found_inf,
      //       tmp_step,
      //       weight_decay,
      //       fp16_weight_decay_end_idx,
      //       beta1,
      //       beta2,
      //       epsilon,
      //       max_global_grad_norm,
      //       rescale_grad);
      //   master_param = fp32_param + fp32_numel;

      int numel = fp16_partial_fused_offsets[fp16_local_param_num] -
                  fp16_partial_fused_offsets[0];
      int weight_decay_end_numel =
          fp16_partial_fused_offsets[fp16_weight_decay_end_idx] -
          fp16_partial_fused_offsets[0];
      printf("fp16 num: %d, fp16_local_param_num: %d\n",
             numel,
             fp16_local_param_num);

      //   int r = xpu::update_lamb_mom_and_trust_ratio_div<xpu_fp16>(
      //       xpu_ctx,
      //       reinterpret_cast<xpu_fp16 *>(master_param + fp16_offset),
      //       fp16_sum_grad,
      //       moment1 + fp32_numel_each_device,
      //       moment2 + fp32_numel_each_device,
      //       trust_ratio_div + fp32_numel_each_device,
      //       1 - beta1pow_cpu,
      //       1 - beta2pow_cpu,
      //       weight_decay,
      //       weight_decay_end_numel,
      //       beta1,
      //       beta2,
      //       epsilon,
      //       update_scale,
      //       numel);

#if 0
      // 正确: 应该用 fp16 grad，fp32 param算 fp32 param
      fp16_sum_grad_tmp = RAII_GUARD.alloc_l3_or_gm<float>(numel);
      int r = xpu::cast<xpu_fp16, float>(
            xpu_ctx,
            fp16_sum_grad,
            fp16_sum_grad_tmp,
            numel);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
#endif
      int r = xpu::update_lamb_mom_and_trust_ratio_div<float, xpu_fp16>(
          xpu_ctx,
          master_param + fp16_offset,
          fp16_sum_grad,
          moment1 + fp32_numel_each_device,
          moment2 + fp32_numel_each_device,
          trust_ratio_div + fp32_numel_each_device,
          1 - beta1pow_cpu,
          1 - beta2pow_cpu,
          weight_decay,
          weight_decay_end_numel,
          beta1,
          beta2,
          epsilon,
          update_scale,
          numel);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "update_lamb_param_and_beta_pows");

      VLOG(10) << "Update FP16 Moment and TrustRatioDiv done";
    }

    VLOG(10) << "Update Moment and TrustRatioDiv done hehahaha";

    // Step 8: calculate L2-Norm square of parameter and trust_ratio_div
    memory::Buffer square_norm_buffer(place);
    auto *param_square_norm = square_norm_buffer.Alloc<float>(2 * param_num);
    auto *trust_ratio_div_square_norm = param_square_norm + param_num;
    if (num_devices > 1) {
      assert(0);
    }
    // 求和不开方，最后统一开方
    MultiTensorL2Norm<float>(dev_ctx,
                             fp32_param,
                             fused_offsets,
                             fp32_global_param_num,
                             param_square_norm);

    if (use_master_param_norm) {
      //   MultiTensorL2Norm(place,
      //                     stream,
      //                     master_param + fp16_offset,
      //                     fp16_partial_fused_offsets,
      //                     fp16_local_param_num,
      //                     param_square_norm + fp16_local_start_idx);

      MultiTensorL2Norm<float>(dev_ctx,
                               master_param + fp16_offset,
                               fp16_partial_fused_offsets,
                               fp16_local_param_num,
                               param_square_norm + fp16_local_start_idx);
    } else {
      assert(0);
    }

    MultiTensorL2Norm<float>(
        dev_ctx,
        trust_ratio_div,
        fp32_partial_fused_offsets,
        fp32_local_param_num,
        trust_ratio_div_square_norm + fp32_local_start_idx);
    MultiTensorL2Norm<float>(
        dev_ctx,
        trust_ratio_div + fp32_numel_each_device,
        fp16_partial_fused_offsets,
        fp16_local_param_num,
        trust_ratio_div_square_norm + fp16_local_start_idx);

    // VLOG(1) << "TrustRatioDiv L2-Norm before allreduce: "
    //         << FlattenToString(trust_ratio_div_square_norm, param_num,
    //         place);
    if (num_devices > 1) {
      assert(0);
    }

    LogParamAndTrustRatioDivSquareNorm<1>(
        ctx, param_square_norm, trust_ratio_div_square_norm);
    VLOG(10) << "Calculate L2-Norm of Param and TrustRatioDiv done";

    // Step 9: update parameter, beta1pow, beta2pow. All gather parameters.
    if (has_fp32_param) {
      //   MultiTensorUpdateLambParamAndBetaPows<float>(
      //       dev_ctx,
      //       fp32_partial_fused_offsets,
      //       fp32_local_param_num,
      //       trust_ratio_div,
      //       lr,
      //       param_square_norm + fp32_local_start_idx,
      //       trust_ratio_div_square_norm + fp32_local_start_idx,
      //       found_inf,
      //       fp32_param + fp32_offset,
      //       nullptr,
      //       beta1pow,
      //       beta2pow,
      //       beta1,
      //       beta2);

      int numel = fp32_partial_fused_offsets[fp32_local_param_num] -
                  fp32_partial_fused_offsets[0];
      printf("fp32 num: %d, fp32_local_param_num: %d, lr: %p\n",
             numel,
             fp32_local_param_num,
             lr);

      //   int r = xpu::update_lamb_param_and_beta_pows<float>(
      //       xpu_ctx,
      //       fp32_param + fp32_offset,
      //       trust_ratio_div,
      //       param_square_norm + fp32_local_start_idx,
      //       trust_ratio_div_square_norm + fp32_local_start_idx,
      //       fp32_param + fp32_offset,
      //       // *lr, //Todo: null
      //       1.0,
      //       numel);
      //   PADDLE_ENFORCE_XDNN_SUCCESS(r, "update_lamb_param_and_beta_pows");
      MultiTensorUpdateLambParamAndBetaPows<float, float>(
          dev_ctx,
          fp32_partial_fused_offsets,
          fp32_local_param_num,
          trust_ratio_div,
          cpu_lr,
          param_square_norm + fp32_local_start_idx,
          trust_ratio_div_square_norm + fp32_local_start_idx,
          fp32_param + fp32_offset,
          nullptr,
          beta1pow,
          beta2pow,
          beta1,
          beta2,
          false);

      if (num_devices > 1) {
        assert(0);
      }

      beta1pow = nullptr;
      beta2pow = nullptr;
    }
    if (has_fp16_param) {
      //   MultiTensorUpdateLambParamAndBetaPows<float16>(
      //       dev_ctx,
      //       fp16_partial_fused_offsets,
      //       fp16_local_param_num,
      //       trust_ratio_div + fp32_numel_each_device,
      //       lr,
      //       param_square_norm + fp16_local_start_idx,
      //       trust_ratio_div_square_norm + fp16_local_start_idx,
      //       found_inf,
      //       fp16_param + fp16_offset,
      //       master_param + fp16_offset,
      //       beta1pow,
      //       beta2pow,
      //       beta1,
      //       beta2);
      int numel = fp16_partial_fused_offsets[fp16_local_param_num] -
                  fp16_partial_fused_offsets[0];
      printf("fp16 num: %d, fp16_local_param_num: %d\n",
             numel,
             fp16_local_param_num);

      // 正确应该是: 用 fp32 的param, 更新到 fp16_param, 和 master_param
      MultiTensorUpdateLambParamAndBetaPows<xpu_fp16, float>(
          dev_ctx,
          fp16_partial_fused_offsets,
          fp16_local_param_num,
          trust_ratio_div + fp32_numel_each_device,
          cpu_lr,
          param_square_norm + fp16_local_start_idx,
          trust_ratio_div_square_norm + fp16_local_start_idx,
          fp16_sum_grad + fp16_offset,
          master_param + fp16_offset,
          beta1pow,
          beta2pow,
          beta1,
          beta2,
          use_master_param_norm);

      if (num_devices > 1) {
        assert(0);
      }
    }
    // Todo: update beta

    VLOG(10) << "Update Param done";

    VLOG(1) << "IsFinite: " << IsFinite(dev_ctx, fp32_square_grad_norm);

    // (void)dev_ctx;
    // (void)place;
    // (void)xpu_ctx;

    (void)fp16_param;

    (void)fp32_param;
    (void)fp32_weight_decay_end_idx;
    (void)fp16_weight_decay_end_idx;
    (void)global_scale;
    (void)lr;
    (void)moment1;
    (void)moment2;
    (void)beta1pow;
    (void)beta2pow;
    (void)found_inf;
    (void)weight_decay;
    (void)beta1;
    (void)beta2;
    (void)epsilon;
    (void)fp32_square_grad_norm;

    (void)step;
    (void)fp32_offset;
    (void)fp16_offset;
    (void)master_param;
  }
};

}  // namespace operators
}  // namespace paddle

namespace plat = paddle::platform;
namespace ops = paddle::operators;

REGISTER_OP_XPU_KERNEL(
    distributed_fused_lamb,
    ops::DistributedFusedLambOpKernel<phi::XPUContext, float>);
