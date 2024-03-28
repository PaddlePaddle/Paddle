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

#include <glog/logging.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/cutlass/fully_connected/fc_decl.h"

#include "paddle/phi/backends/dynload/cutlass_fc.h"
#include <iostream>
#include "paddle/phi/kernels/funcs/common_shape.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

typedef void (*func)(phi::fusion::cutlass_internal::FcAllParams);

template <typename T, typename Context>
void FCKernel(const Context& dev_ctx,
              const DenseTensor& input,
              const DenseTensor& w,
              const paddle::optional<DenseTensor>& bias,
              const int in_num_col_dims,
              const std::string& activation_type,
              const bool padding_weights,
              DenseTensor* out) {
  // std::cout << "kai-----------FusedFcAddActKernel.begin: " << std::endl;

  // 暂时把下两个参数从参数列表移到这里, 以对齐FCKernel
  // const std::string& data_format,
  // float leaky_alpha,
  const std::string data_format("RRR");
  float leaky_alpha = 0.01;

  if(!bias){
    PADDLE_THROW(phi::errors::InvalidArgument("bias is needed!!!"));
    return;
  }
  PADDLE_ENFORCE_EQ(padding_weights, false,
                    phi::errors::PermissionDenied("Weight padding in fc can not be used in GPU scope."));

  auto weight_dims = w.dims();
  /// fc_out should be reshape when run since can not get lod in infershap
  std::vector<int64_t> output_dims;
  phi::funcs::FCOutputSize(
      input.dims(), weight_dims, output_dims, in_num_col_dims, padding_weights);
  out->Resize(common::make_ddim(output_dims));
  out->set_lod(input.lod());
  ///

  dev_ctx.template Alloc<T>(out);
  auto bias_dims = bias->dims();
  auto out_dims = out->dims();

  CHECK_EQ(weight_dims.size() == 2UL, true);
  CHECK_EQ(data_format == "RRR", true);

  /// 这里参考blas的实现
  int M = common::product(out_dims) / weight_dims[1];
  const int K = weight_dims[0];
  const int N = weight_dims[1];
  
  bool vecBias = true;
  if(bias_dims.size()>2){
    vecBias = false;
  }
  else if(bias_dims.size()==2 && bias_dims[0] != 1){
    vecBias = false;
  }
  
  // std::cout << "MNK: [" << M  << ", " << N << ", " << K << "]" << std::endl;
  // 在RRR的情况下
  const int lda = K;
  const int ldb = N;
  const int ldd = N;
  // 这里暂时写死GEMM内核实现部分的Align参数为8，判定一下输入是否可接受
  const int alignA = 8;
  const int alignB = 8;
  bool isAMisaligned = false;
  bool isBMisaligned = false;
  // 在RRR情况下 
  isAMisaligned = lda % alignA;
  isBMisaligned = ldb % alignB;
  if (isAMisaligned) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "  returning kErrorMisalignedOperand for input operand"));
    return;
  }
  if (isBMisaligned) {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "  returning kErrorMisalignedOperand for weight operand"));
    return;
  }

  int64_t device_id = dev_ctx.GetPlace().GetDeviceId();
  int sm_version = backends::gpu::GetGPUComputeCapability(device_id);

  auto get_fc_dtype = [&](decltype(input.dtype()) input_type)
      -> phi::fusion::cutlass_internal::FcDataType {
    switch (input_type) {
      case phi::DataType::FLOAT32:
        return FcDataType::fp32;
      case phi::DataType::FLOAT16:
        return FcDataType::fp16;
      case phi::DataType::BFLOAT16:
        return FcDataType::bf16;
    }
  };

  auto cutlass_dispatch_sm_version = [&](int device_sm_version) -> int {
    if (device_sm_version < 75) {
      PADDLE_ENFORCE_GE(
          device_sm_version,
          75,
          phi::errors::PreconditionNotMet(
              "fused_fc_add_act only supports sm >= 75, but got %d.",
              device_sm_version));
    } else if (device_sm_version > 80) {
      return 80;
    } else {
      return device_sm_version;
    }
  };

  FcAllParams params = {
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
      get_fc_dtype(input.dtype()),
      vecBias,
      cutlass_dispatch_sm_version(sm_version),
      leaky_alpha,       // for leaky_relu
  };

  void* dlhandler = phi::dynload::GetCutlassFcHandle();
  func fc_func = NULL;
  CHECK_EQ(dlhandler == NULL, false);
  // std::cout << "kai-----------FusedFcAddActKernel.activation_type: " << activation_type << std::endl;
  if (activation_type == "relu") {
    fc_func = (func)(dlsym(dlhandler, "FcBiasRelu"));
  } else if (activation_type == "swish") {
    fc_func = (func)(dlsym(dlhandler, "FcBiasSilu"));
  } else if (activation_type == "identity" || activation_type == "") {
    fc_func = (func)(dlsym(dlhandler, "FcBias"));
  } else if (activation_type == "leaky_relu") {
    fc_func = (func)(dlsym(dlhandler, "FcBiasLeakyRelu"));
  } else if (activation_type == "sigmoid") {
    fc_func = (func)(dlsym(dlhandler, "FcBiasSigmoid"));
  } else if (activation_type == "gelu"){
    fc_func = (func)(dlsym(dlhandler, "FcBiasGelu"));
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Cutlass does not support this activation_type: %s.", activation_type.c_str()));
  }
  fc_func(params);
}
}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(gemm_epilogue,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::FCKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}