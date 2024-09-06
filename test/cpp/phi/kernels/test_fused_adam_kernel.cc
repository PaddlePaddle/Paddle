// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/generator.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/gpu/gpu_context.h"
#endif

#include "gtest/gtest.h"

#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/infermeta/multiary.h"
#include "paddle/phi/kernels/abs_kernel.h"
#include "paddle/phi/kernels/adam_kernel.h"
#include "paddle/phi/kernels/adamw_kernel.h"
#include "paddle/phi/kernels/cast_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/fused_adam_kernel.h"
#include "paddle/phi/kernels/gaussian_kernel.h"
#include "paddle/phi/kernels/legacy/reduce_max_kernel.h"

namespace phi {

template <typename T, typename Context>
auto GenerateRandomTensorVectors(
    const Context &ctx, const std::vector<std::vector<int64_t>> &shapes) {
  size_t n = shapes.size();
  std::vector<DenseTensor> tensors(n);
  for (size_t i = 0; i < n; ++i) {
    GaussianKernel<T, Context>(ctx,
                               shapes[i],
                               0.0f,
                               1.0f,
                               0,
                               phi::CppTypeToDataType<T>::Type(),
                               &tensors[i]);
  }
  return tensors;
}

template <typename T, typename Context>
auto GenerateConstantTensorVectors(
    const Context &ctx,
    const std::vector<std::vector<int64_t>> &shapes,
    T value) {
  size_t n = shapes.size();
  std::vector<DenseTensor> tensors(n);
  for (size_t i = 0; i < n; ++i) {
    FullKernel<T, Context>(
        ctx, shapes[i], value, phi::CppTypeToDataType<T>::Type(), &tensors[i]);
  }
  return tensors;
}

static auto ToConstTensorPtrVector(const std::vector<DenseTensor> &tensors) {
  std::vector<const DenseTensor *> results;
  results.reserve(tensors.size());
  for (const auto &t : tensors) {
    results.push_back(&t);
  }
  return results;
}

static auto ToMutableTensorPtrVector(
    std::vector<DenseTensor> &tensors) {  // NOLINT
  std::vector<DenseTensor *> results;
  results.reserve(tensors.size());
  for (auto &t : tensors) {
    results.push_back(&t);
  }
  return results;
}

static auto ToMetaTensorVector(const std::vector<DenseTensor> &tensors) {
  std::vector<MetaTensor> results;
  results.reserve(tensors.size());
  for (auto &t : tensors) {
    results.emplace_back(t);
  }
  return results;
}

static auto ToConstMetaTensorPtrVector(
    const std::vector<MetaTensor> &meta_tensors) {
  std::vector<const MetaTensor *> results;
  results.reserve(meta_tensors.size());
  for (auto &t : meta_tensors) {
    results.push_back(&t);
  }
  return results;
}

static auto ToMutableMetaTensorPtrVector(
    std::vector<MetaTensor> &meta_tensors) {  // NOLINT
  std::vector<MetaTensor *> results;
  results.reserve(meta_tensors.size());
  for (auto &t : meta_tensors) {
    results.push_back(&t);
  }
  return results;
}

template <typename T, typename Context>
struct AdamInfo {
  const Context *ctx;
  std::vector<std::vector<int64_t>> shapes;

  std::vector<DenseTensor> params;
  std::vector<DenseTensor> master_params;
  std::vector<DenseTensor> moment1s;
  std::vector<DenseTensor> moment2s;
  std::vector<DenseTensor> moment2s_max;
  std::vector<DenseTensor> beta1_pows;
  std::vector<DenseTensor> beta2_pows;
  DenseTensor learning_rate;
  float beta1;
  float beta2;
  float weight_decay;
  float epsilon = 1e-6;
  bool multi_precision;
  bool use_adamw;
  int chunk_size = 4096;
  bool amsgrad;

  using MT = typename phi::dtype::MPTypeTrait<T>::Type;

  AdamInfo(const Context &ctx_ref,
           const std::vector<std::vector<int64_t>> &shapes,
           float beta1,
           float beta2,
           float weight_decay,
           bool multi_precision,
           bool use_adamw,
           bool amsgrad)
      : ctx(&ctx_ref),
        shapes(shapes),
        beta1(beta1),
        beta2(beta2),
        weight_decay(weight_decay),
        multi_precision(multi_precision),
        use_adamw(use_adamw),
        amsgrad(amsgrad) {
    std::vector<std::vector<int64_t>> one_shapes(shapes.size(),
                                                 std::vector<int64_t>(1, 1));
    std::vector<std::vector<int64_t>> learning_rate_shapes(
        one_shapes.begin(), one_shapes.begin() + 1);

    params = GenerateRandomTensorVectors<T, Context>(*ctx, shapes);
    learning_rate = GenerateConstantTensorVectors<MT, Context>(
        *ctx, learning_rate_shapes, 1e-3)[0];
    moment1s = GenerateConstantTensorVectors<MT, Context>(*ctx, shapes, 0);
    moment2s = GenerateConstantTensorVectors<MT, Context>(*ctx, shapes, 0);
    moment2s_max = GenerateConstantTensorVectors<MT, Context>(*ctx, shapes, 0);

    if (multi_precision) {
      master_params.resize(shapes.size());
      for (size_t i = 0; i < shapes.size(); ++i) {
        master_params[i] = Cast<T, Context>(
            *ctx, params[i], phi::CppTypeToDataType<MT>::Type());
      }
    }

    beta1_pows =
        GenerateConstantTensorVectors<MT, Context>(*ctx, one_shapes, beta1);
    beta2_pows =
        GenerateConstantTensorVectors<MT, Context>(*ctx, one_shapes, beta2);
  }

  void Update(bool use_fused, const std::vector<DenseTensor> &grads) {
    if (use_fused) {
      UpdateWithFusedAdam(grads);
    } else {
      for (size_t j = 0; j < params.size(); ++j) {
        if (use_adamw) {
          UpdateWithAdamWBaseline(grads, j);
        } else {
          UpdateWithAdamBaseline(grads, j);
        }
      }
    }
  }

  static AdamInfo<T, Context> DeepCopy(const AdamInfo &other) {
    AdamInfo copied(*other.ctx,
                    other.shapes,
                    other.beta1,
                    other.beta2,
                    other.weight_decay,
                    other.multi_precision,
                    other.use_adamw,
                    other.amsgrad);
    auto copy_tensor = [&other](const DenseTensor &x, DenseTensor *y) {
      Copy<Context>(*other.ctx, x, x.place(), false, y);
    };

    auto copy_tensors = [&other](const std::vector<DenseTensor> &xs,
                                 std::vector<DenseTensor> *ys) {
      for (size_t i = 0; i < xs.size(); ++i) {
        Copy<Context>(*other.ctx, xs[i], xs[i].place(), false, &((*ys)[i]));
      }
    };

    copy_tensors(other.params, &copied.params);
    copy_tensors(other.master_params, &copied.master_params);
    copy_tensors(other.moment1s, &copied.moment1s);
    copy_tensors(other.moment2s, &copied.moment2s);
    copy_tensors(other.moment2s_max, &copied.moment2s_max);
    copy_tensors(other.beta1_pows, &copied.beta1_pows);
    copy_tensors(other.beta2_pows, &copied.beta2_pows);
    copy_tensor(other.learning_rate, &copied.learning_rate);
    copied.epsilon = other.epsilon;
    copied.chunk_size = other.chunk_size;
    other.ctx->Wait();
    return copied;
  }

 private:
  void UpdateWithFusedAdam(const std::vector<DenseTensor> &grads) {
    auto param_metas = ToMetaTensorVector(params);
    auto grad_metas = ToMetaTensorVector(grads);
    auto master_param_metas = ToMetaTensorVector(master_params);
    auto moment1_metas = ToMetaTensorVector(moment1s);
    auto moment2_metas = ToMetaTensorVector(moment2s);
    auto moment2_max_metas = ToMetaTensorVector(moment2s_max);
    auto beta1_pow_metas = ToMetaTensorVector(beta1_pows);
    auto beta2_pow_metas = ToMetaTensorVector(beta2_pows);

    FusedAdamInferMeta(ToConstMetaTensorPtrVector(param_metas),
                       ToConstMetaTensorPtrVector(grad_metas),
                       learning_rate,
                       ToConstMetaTensorPtrVector(moment1_metas),
                       ToConstMetaTensorPtrVector(moment2_metas),
                       ToConstMetaTensorPtrVector(moment2_max_metas),
                       ToConstMetaTensorPtrVector(beta1_pow_metas),
                       ToConstMetaTensorPtrVector(beta2_pow_metas),
                       multi_precision
                           ? paddle::make_optional(
                                 ToConstMetaTensorPtrVector(master_param_metas))
                           : paddle::none,
                       MetaTensor(),
                       beta1,
                       beta2,
                       epsilon,
                       chunk_size,
                       weight_decay,
                       use_adamw,
                       multi_precision,
                       false,
                       amsgrad,
                       ToMutableMetaTensorPtrVector(param_metas),
                       ToMutableMetaTensorPtrVector(moment1_metas),
                       ToMutableMetaTensorPtrVector(moment2_metas),
                       ToMutableMetaTensorPtrVector(moment2_max_metas),
                       ToMutableMetaTensorPtrVector(beta1_pow_metas),
                       ToMutableMetaTensorPtrVector(beta2_pow_metas),
                       ToMutableMetaTensorPtrVector(master_param_metas));

    FusedAdamKernel<T, Context>(
        *ctx,
        ToConstTensorPtrVector(params),
        ToConstTensorPtrVector(grads),
        learning_rate,
        ToConstTensorPtrVector(moment1s),
        ToConstTensorPtrVector(moment2s),
        ToConstTensorPtrVector(moment2s_max),
        ToConstTensorPtrVector(beta1_pows),
        ToConstTensorPtrVector(beta2_pows),
        multi_precision
            ? paddle::make_optional(ToConstTensorPtrVector(master_params))
            : paddle::none,
        paddle::none,
        beta1,
        beta2,
        epsilon,
        chunk_size,
        weight_decay,
        use_adamw,
        multi_precision,
        false,
        amsgrad,
        ToMutableTensorPtrVector(params),
        ToMutableTensorPtrVector(moment1s),
        ToMutableTensorPtrVector(moment2s),
        ToMutableTensorPtrVector(moment2s_max),
        ToMutableTensorPtrVector(beta1_pows),
        ToMutableTensorPtrVector(beta2_pows),
        ToMutableTensorPtrVector(master_params));
  }

  void UpdateWithAdamWBaseline(const std::vector<DenseTensor> &grads,
                               size_t idx) {
    AdamwDenseKernel<T, Context>(
        *ctx,
        params[idx],
        grads[idx],
        learning_rate,
        moment1s[idx],
        moment2s[idx],
        moment2s_max[idx],
        beta1_pows[idx],
        beta2_pows[idx],
        multi_precision ? paddle::make_optional(master_params[idx])
                        : paddle::none,
        paddle::none,
        beta1,
        beta2,
        epsilon,
        1.0,
        weight_decay,
        true,
        false,
        1000,
        multi_precision,
        false,
        amsgrad,
        &params[idx],
        &moment1s[idx],
        &moment2s[idx],
        &moment2s_max[idx],
        &beta1_pows[idx],
        &beta2_pows[idx],
        multi_precision ? &master_params[idx] : nullptr);
  }

  void UpdateWithAdamBaseline(const std::vector<DenseTensor> &grads,
                              size_t idx) {
    AdamDenseKernel<T, Context>(
        *ctx,
        params[idx],
        grads[idx],
        learning_rate,
        moment1s[idx],
        moment2s[idx],
        moment2s_max[idx],
        beta1_pows[idx],
        beta2_pows[idx],
        multi_precision ? paddle::make_optional(master_params[idx])
                        : paddle::none,
        paddle::none,
        beta1,
        beta2,
        epsilon,
        false,
        1000,
        multi_precision,
        false,
        amsgrad,
        &params[idx],
        &moment1s[idx],
        &moment2s[idx],
        &moment2s_max[idx],
        &beta1_pows[idx],
        &beta2_pows[idx],
        multi_precision ? &master_params[idx] : nullptr);
  }
};

template <typename T, typename Context>
auto MaxDiff(const Context &ctx,
             const DenseTensor &x_t,
             const DenseTensor &y_t) {
  using MT = typename AdamInfo<T, Context>::MT;
  auto mp_dtype = phi::CppTypeToDataType<MT>::Type();
  auto x = Cast<T, Context>(ctx, x_t, mp_dtype);
  auto y = Cast<T, Context>(ctx, y_t, mp_dtype);

  EXPECT_EQ(x.dims(), y.dims());
  DenseTensor diff, diff_reduced, diff_reduced_cpu;

  diff.Resize(x.dims());
  ctx.template Alloc<MT>(&diff);
  SubtractKernel<MT, Context>(ctx, x, y, &diff);
  AbsKernel<MT, Context>(ctx, diff, &diff);

  diff_reduced.Resize({1});
  ctx.template Alloc<MT>(&diff_reduced);
  MaxRawKernel<MT, Context>(ctx,
                            diff,
                            common::vectorize<int64_t>(x.dims()),
                            false,
                            true,
                            &diff_reduced);

  diff_reduced_cpu.Resize(diff_reduced.dims());
  ctx.template HostAlloc<MT>(&diff_reduced_cpu);
  Copy<Context>(ctx, diff_reduced, CPUPlace(), true, &diff_reduced_cpu);
  EXPECT_EQ(diff_reduced_cpu.place(), CPUPlace());
  return diff_reduced_cpu.data<MT>()[0];
}

template <typename T, typename Context>
auto MaxDiff(const Context &ctx,
             const std::vector<DenseTensor> &xs,
             const std::vector<DenseTensor> &ys) {
  using MT = typename AdamInfo<T, Context>::MT;
  MT diff = 0;
  for (size_t i = 0; i < xs.size(); ++i) {
    diff = std::max<MT>(diff, MaxDiff<T, Context>(ctx, xs[i], ys[i]));
  }
  return diff;
}

template <typename T, typename PlaceType>
void TestFusedAdamBase(const std::vector<std::vector<int64_t>> &shapes,
                       float atol,
                       bool use_adamw,
                       bool amsgrad,
                       bool multi_precision = false,
                       float beta1 = 0.9,
                       float beta2 = 0.99,
                       float weight_decay = 0.1,
                       size_t steps = 5,
                       uint64_t seed = 10) {
  const auto &ctx = *phi::DeviceContextPool::Instance().GetByPlace(PlaceType());
  using Context = typename std::remove_const<
      typename std::remove_pointer<decltype(&ctx)>::type>::type;
  ctx.GetGenerator()->SetCurrentSeed(seed);
  AdamInfo<T, Context> info1(ctx,
                             shapes,
                             beta1,
                             beta2,
                             weight_decay,
                             multi_precision,
                             use_adamw,
                             amsgrad);
  auto info2 = AdamInfo<T, Context>::DeepCopy(info1);

  for (size_t i = 0; i < steps; ++i) {
    auto grads = GenerateRandomTensorVectors<T>(ctx, shapes);
    info1.Update(false, grads);
    info2.Update(true, grads);
  }

  using MT = typename decltype(info1)::MT;

#define PD_ADAM_TEST_COMP(__field, __dtype)                          \
  do {                                                               \
    MT __diff = MaxDiff<__dtype>(ctx, info1.__field, info2.__field); \
    EXPECT_LE(__diff, static_cast<MT>(atol))                         \
        << #__field << " has diff when use_adamw = " << use_adamw    \
        << " , multi_precision = " << multi_precision;               \
  } while (0)

  PD_ADAM_TEST_COMP(beta1_pows, MT);
  PD_ADAM_TEST_COMP(beta2_pows, MT);
  PD_ADAM_TEST_COMP(params, T);
  PD_ADAM_TEST_COMP(master_params, MT);
  PD_ADAM_TEST_COMP(moment1s, MT);
  PD_ADAM_TEST_COMP(moment2s, MT);
  PD_ADAM_TEST_COMP(moment2s_max, MT);
}

static auto GenerateRandomShapes(size_t n, uint64_t low, uint64_t high) {
  std::random_device device;
  std::default_random_engine engine(device());
  std::uniform_int_distribution<uint64_t> dist(low, high);
  std::vector<std::vector<int64_t>> shapes(n);
  for (size_t i = 0; i < n; ++i) {
    shapes[i].push_back(static_cast<int64_t>(dist(engine)));
  }
  return shapes;
}

TEST(fused_adam, test_fp32_cpu) {
  auto shapes = GenerateRandomShapes(30, 10, 20);
  float atol = 0.0f;
  for (auto use_adamw : {false, true}) {
    for (auto amsgrad : {false, true}) {
      TestFusedAdamBase<float, CPUPlace>(shapes, atol, use_adamw, amsgrad);
    }
  }
}

#ifdef PADDLE_WITH_CUDA
TEST(fused_adam, test_fp32_gpu) {
  auto shapes = GenerateRandomShapes(40, 0, 2 << 18);
  float atol = 0.0f;
  for (auto use_adamw : {false, true}) {
    for (auto amsgrad : {false, true}) {
      TestFusedAdamBase<float, GPUPlace>(shapes, atol, use_adamw, amsgrad);
    }
  }
}

TEST(fused_adam, test_fp16_gpu) {
  auto shapes = GenerateRandomShapes(40, 0, 2 << 18);
  float atol = 5e-3f;
  for (auto use_adamw : {false, true}) {
    for (auto amsgrad : {false, true}) {
      TestFusedAdamBase<dtype::float16, GPUPlace>(
          shapes, atol, use_adamw, amsgrad, true);
    }
  }
}
#endif

}  // namespace phi
