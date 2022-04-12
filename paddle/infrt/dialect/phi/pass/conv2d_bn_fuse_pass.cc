// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include "paddle/infrt/backends/host/phi_allocator.h"
#include "paddle/infrt/backends/host/phi_context.h"
#include "paddle/infrt/dialect/pd/ir/pd_ops.h"
#include "paddle/infrt/dialect/phi/pass/utils/attribute_helper.h"

namespace infrt {
namespace dialect {
namespace {
void compute_alpha(const ::phi::CPUContext& context,
                   const ::phi::DenseTensor& variance,
                   const ::phi::DenseTensor& scale,
                   const float epsilon,
                   ::phi::DenseTensor* alpha) {
  const auto* variance_p = variance.data<float>();
  const auto* scale_p = scale.data<float>();
  alpha->set_meta(variance.meta());
  alpha->AllocateFrom(const_cast<::phi::Allocator*>(&context.GetAllocator()),
                      alpha->dtype());
  auto* alpha_p = alpha->data<float>();
  for (int i = 0; i < variance.numel(); ++i) {
    alpha_p[i] = scale_p[i] / std::sqrt(variance_p[i] + epsilon);
  }
}

void compute_beta(const ::phi::CPUContext& context,
                  const ::phi::DenseTensor& alpha,
                  const ::phi::DenseTensor& mean,
                  ::phi::DenseTensor* beta) {
  const auto* mean_p = mean.data<float>();
  const auto* alpha_p = alpha.data<float>();
  beta->set_meta(alpha.meta());
  beta->AllocateFrom(const_cast<::phi::Allocator*>(&context.GetAllocator()),
                     beta->dtype());
  auto* beta_p = beta->data<float>();
  for (int i = 0; i < alpha.numel(); ++i) {
    beta_p[i] = (-mean_p[i]) * alpha_p[i];
  }
}

void compute_filter(const ::phi::CPUContext& context,
                    const ::phi::DenseTensor& filter,
                    const ::phi::DenseTensor& alpha,
                    ::phi::DenseTensor* filter_out) {
  CHECK_EQ(filter.dims().size(), 4);
  CHECK_EQ(alpha.dims().size(), 1);
  CHECK_EQ(filter.dims()[0], alpha.dims()[0]);
  filter_out->set_meta(filter.meta());
  filter_out->AllocateFrom(
      const_cast<::phi::Allocator*>(&context.GetAllocator()),
      filter_out->dtype());
  const auto* filter_d = filter.data<float>();
  const auto* alpha_d = alpha.data<float>();
  auto* filter_out_d = filter_out->data<float>();
  int w = filter.dims()[1] * filter.dims()[2] * filter.dims()[3];
  for (int i = 0; i < filter.dims()[0]; ++i) {
    for (int j = 0; j < w; ++j) {
      filter_out_d[i * w + j] = filter_d[i * w + j] * alpha_d[i];
    }
  }
}

void compute_bias(const ::phi::CPUContext& context,
                  const ::phi::DenseTensor& bias,
                  const ::phi::DenseTensor& beta,
                  ::phi::DenseTensor* bias_out) {
  bias_out->set_meta(bias.meta());
  bias_out->AllocateFrom(const_cast<::phi::Allocator*>(&context.GetAllocator()),
                         bias_out->dtype());
  const auto* bias_d = bias.data<float>();
  const auto* beta_d = beta.data<float>();
  auto* bias_out_d = bias_out->data<float>();
  for (int i = 0; i < bias.numel(); ++i) {
    bias_out_d[i] = bias_d[i] + beta_d[i];
  }
}

}  // namespace

::mlir::Value Conv2dBnFuse_CreateFilter(
    ::mlir::PatternRewriter& rewriter,  // NOLINT
    ::mlir::Location& loc,              // NOLINT
    ::mlir::Value filter_attr,
    ::mlir::Value variance_attr,
    ::mlir::Value scale_attr,
    ::mlir::FloatAttr epsilon_attr) {
  ::infrt::backends::CpuPhiContext cpu_context;
  auto* cpu_allocator =
      const_cast<::phi::Allocator*>(&cpu_context.GetAllocator());
  auto filter =
      CreateDenseTensorFromWeightOp(cpu_allocator, filter_attr.getDefiningOp());
  auto variance = CreateDenseTensorFromWeightOp(cpu_allocator,
                                                variance_attr.getDefiningOp());
  auto scale =
      CreateDenseTensorFromWeightOp(cpu_allocator, scale_attr.getDefiningOp());
  float epsilon = epsilon_attr.cast<mlir::FloatAttr>().getValueAsDouble();

  ::phi::DenseTensor alpha, filter_out;
  compute_alpha(cpu_context, variance, scale, epsilon, &alpha);
  compute_filter(cpu_context, filter, alpha, &filter_out);

  auto context_op = CreateCPUContextOp(rewriter, loc);
  auto weight_op = CreateWeightOpFromDenseTensor(
      rewriter, loc, context_op.output(), filter_out);
  return weight_op.output();
}

::mlir::Value Conv2dBnFuse_CreateBias(
    ::mlir::PatternRewriter& rewriter,  // NOLINT
    ::mlir::Location& loc,              // NOLINT
    ::mlir::Value mean_attr,
    ::mlir::Value variance_attr,
    ::mlir::Value scale_attr,
    ::mlir::Value bias_attr,
    ::mlir::FloatAttr epsilon_attr) {
  ::infrt::backends::CpuPhiContext cpu_context;
  auto* cpu_allocator =
      const_cast<::phi::Allocator*>(&cpu_context.GetAllocator());
  auto mean =
      CreateDenseTensorFromWeightOp(cpu_allocator, mean_attr.getDefiningOp());
  auto variance = CreateDenseTensorFromWeightOp(cpu_allocator,
                                                variance_attr.getDefiningOp());
  auto scale =
      CreateDenseTensorFromWeightOp(cpu_allocator, scale_attr.getDefiningOp());
  auto bias =
      CreateDenseTensorFromWeightOp(cpu_allocator, bias_attr.getDefiningOp());
  float epsilon = epsilon_attr.cast<mlir::FloatAttr>().getValueAsDouble();

  ::phi::DenseTensor alpha, beta, bias_out;
  compute_alpha(cpu_context, variance, scale, epsilon, &alpha);
  compute_beta(cpu_context, alpha, mean, &beta);
  compute_bias(cpu_context, bias, beta, &bias_out);

  auto context_op = CreateCPUContextOp(rewriter, loc);
  auto weight_op = CreateWeightOpFromDenseTensor(
      rewriter, loc, context_op.output(), bias_out);
  return weight_op.output();
}

}  // namespace dialect
}  // namespace infrt

#include "paddle/infrt/dialect/phi/pass/phi_op_fuse.cpp.inc"

namespace {
struct Conv2dBnFusePass
    : public mlir::PassWrapper<Conv2dBnFusePass, mlir::FunctionPass> {
 public:
  ::llvm::StringRef getName() const override { return "Conv2dBnFusePass"; }

  llvm::StringRef getArgument() const override { return "conv2d-bn-fuse"; }

  void runOnFunction() override;
};

// Implementation of the Conv2dBnFusePass.
void Conv2dBnFusePass::runOnFunction() {
  ::mlir::RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  if (::mlir::failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}
}  // namespace

mlir::PassRegistration<Conv2dBnFusePass> conv2d_bn_fuse_pass;
