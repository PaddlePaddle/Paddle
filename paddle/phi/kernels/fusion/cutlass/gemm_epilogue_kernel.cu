// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/cutlass/gemm_epilogue/gemm_epilogue_decl.h"

#include "paddle/phi/backends/dynload/cutlass_gemm_epilogue.h"
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

typedef void (*func)(phi::fusion::cutlass_internal::GemmEpilogueAllParams);

template <typename T, typename Context>
void GemmEpilogueKernel(const Context& dev_ctx,
                        const DenseTensor& input,
                        const DenseTensor& w,
                        const paddle::optional<DenseTensor>& bias,
                        const int in_num_col_dims,
                        const std::string& activation_type,
                        const bool padding_weights,
                        DenseTensor* out) {
  if (!bias) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "In gemm_epilogue kernel, bias should be provided."));
    return;
  }
  PADDLE_ENFORCE_EQ(
      padding_weights,
      false,
      common::errors::PermissionDenied(
          "Weight padding in gemm_epilogue can not be used in GPU scope."));

  auto weight_dims = w.dims();
  PADDLE_ENFORCE_EQ(weight_dims.size(),
                    2UL,
                    common::errors::InvalidArgument(
                        "In gemm_epilogue kernel, weight_dims should be 2."));
  // gemm_epilogue_out should be reshape since can not get lod in infershape
  std::vector<int64_t> output_dims;
  phi::funcs::FCOutputSize(
      input.dims(), weight_dims, output_dims, in_num_col_dims, padding_weights);
  out->Resize(common::make_ddim(output_dims));
  out->set_lod(input.lod());

  dev_ctx.template Alloc<T>(out);

  auto out_dims = out->dims();
  auto bias_dims = bias->dims();
  // This Op supports two types of bias elementwiseAdd.
  // For an output of dimension [M,N], a bias is either a vector of length N or
  // also a matrix of dimension [M,N]. isVec_bias is used to distinguish whether
  // bias is a vector or not
  bool isVec_bias = true;
  if (bias_dims.size() > 2 || (bias_dims.size() == 2 && bias_dims[0] != 1)) {
    isVec_bias = false;
  }

  int M = common::product(out_dims) / weight_dims[1];
  const int K = weight_dims[0];
  const int N = weight_dims[1];
  const int lda = K;
  const int ldb = N;
  const int ldd = N;
  const int alignA = 8;
  const int alignB = 8;
  bool isAMisaligned = false;
  bool isBMisaligned = false;
  isAMisaligned = lda % alignA;
  isBMisaligned = ldb % alignB;
  if (isAMisaligned) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "  returning kErrorMisalignedOperand for input operand"));
    return;
  }
  if (isBMisaligned) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "  returning kErrorMisalignedOperand for weight operand"));
    return;
  }

  int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
  int sm_version = backends::gpu::GetGPUComputeCapability(device_id);

  auto get_gemm_epilogue_dtype = [&](decltype(input.dtype()) input_type)
      -> phi::fusion::cutlass_internal::GemmEpilogueDataType {
    switch (input_type) {
      case phi::DataType::FLOAT32:
        return GemmEpilogueDataType::fp32;
      case phi::DataType::FLOAT16:
        return GemmEpilogueDataType::fp16;
      case phi::DataType::BFLOAT16:
        return GemmEpilogueDataType::bf16;
      default:
        return GemmEpilogueDataType::fp32;
    }
  };
  auto gemm_epilogue_dtype = get_gemm_epilogue_dtype(input.dtype());
  if ((gemm_epilogue_dtype != GemmEpilogueDataType::fp16) &&
      (gemm_epilogue_dtype != GemmEpilogueDataType::bf16)) {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Gemm_epilogue kernel only supports fp16 "
        "and bf16, input dtype error!"));
    return;
  }

  auto cutlass_dispatch_sm_version = [&](int device_sm_version) -> int {
    if (device_sm_version < 80) {
      PADDLE_ENFORCE_GE(device_sm_version,
                        80,
                        common::errors::PreconditionNotMet(
                            "Gemm_epilogue only supports sm >= 80, but got %d.",
                            device_sm_version));
    } else if (device_sm_version > 80) {
      return 80;
    } else {
      return device_sm_version;
    }
  };

  void* workspace = nullptr;
  size_t workspace_size_bytes =
      ((M - 1 + 16) / 16) * ((N - 1 + 64) / 64) * sizeof(int);
  phi::Allocator::AllocationPtr tmp_ptr = phi::memory_utils::Alloc(
      dev_ctx.GetPlace(),
      workspace_size_bytes,
      phi::Stream(reinterpret_cast<phi::StreamId>(dev_ctx.stream())));
  workspace = tmp_ptr->ptr();

  GemmEpilogueAllParams params = {
      reinterpret_cast<const void*>(input.data<T>()),
      reinterpret_cast<const void*>(w.data<T>()),
      reinterpret_cast<const void*>(bias->data<T>()),
      reinterpret_cast<void*>(out->data<T>()),
      M,
      N,
      K,
      lda,
      ldb,
      ldd,
      dev_ctx.stream(),
      gemm_epilogue_dtype,
      isVec_bias,
      cutlass_dispatch_sm_version(sm_version),
      0.01,  // for leaky_relu
      workspace,
  };

  void* dlhandler = phi::dynload::GetCutlassGemmEpilogueHandle();
  func gemm_epilogue_func = NULL;
  PADDLE_ENFORCE_NOT_NULL(
      dlhandler,
      common::errors::PreconditionNotMet(
          "CutlassGemmEpilogueHandle should not be NULL, dynload error."));
  if (activation_type == "identity" || activation_type == "") {
    gemm_epilogue_func = (func)(dlsym(dlhandler, "MatmulAdd"));
  } else if (activation_type == "relu") {
    gemm_epilogue_func = (func)(dlsym(dlhandler, "MatmulAddRelu"));
  } else if (activation_type == "gelu") {
    gemm_epilogue_func = (func)(dlsym(dlhandler, "MatmulAddGelu"));
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Cutlass does not support this activation_type: %s.",
        activation_type.c_str()));
  }
  /// these three acts are not supported by pass now and are commented out to
  /// prevent the lib.so too large.
  // else if (activation_type == "leaky_relu") {
  //   gemm_epilogue_func = (func)(dlsym(dlhandler, "MatmulAddLeakyRelu"));
  // }
  // else if (activation_type == "sigmoid") {
  //   gemm_epilogue_func = (func)(dlsym(dlhandler, "MatmulAddSigmoid"));
  // }
  // else if (activation_type == "swish") {
  //   gemm_epilogue_func = (func)(dlsym(dlhandler, "MatmulAddSilu"));
  // }

  gemm_epilogue_func(params);
}

}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(gemm_epilogue,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::GemmEpilogueKernel,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
