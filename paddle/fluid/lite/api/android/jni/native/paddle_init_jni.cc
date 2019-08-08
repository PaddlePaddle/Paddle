/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/lite/api/android/jni/native/paddle_init_jni.h"

#include <memory>

#include "paddle/fluid/lite/api/paddle_lite_factory_helper.h"
#include "paddle/fluid/lite/api/paddle_use_kernels.h"
#include "paddle/fluid/lite/api/paddle_use_ops.h"
#include "paddle/fluid/lite/api/paddle_use_passes.h"
#include "paddle/fluid/lite/kernels/arm/activation_compute.h"
#include "paddle/fluid/lite/kernels/arm/batch_norm_compute.h"
#include "paddle/fluid/lite/kernels/arm/calib_compute.h"
#include "paddle/fluid/lite/kernels/arm/concat_compute.h"
#include "paddle/fluid/lite/kernels/arm/conv_compute.h"
#include "paddle/fluid/lite/kernels/arm/dropout_compute.h"
#include "paddle/fluid/lite/kernels/arm/elementwise_compute.h"
#include "paddle/fluid/lite/kernels/arm/fc_compute.h"
#include "paddle/fluid/lite/kernels/arm/mul_compute.h"
#include "paddle/fluid/lite/kernels/arm/pool_compute.h"
#include "paddle/fluid/lite/kernels/arm/scale_compute.h"
#include "paddle/fluid/lite/kernels/arm/softmax_compute.h"
#include "paddle/fluid/lite/kernels/arm/split_compute.h"
#include "paddle/fluid/lite/kernels/arm/transpose_compute.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ARM_KERNEL_POINTER(kernel_class_name__)                    \
  std::unique_ptr<paddle::lite::kernels::arm::kernel_class_name__> \
      p##kernel_class_name__(                                      \
          new paddle::lite::kernels::arm::kernel_class_name__);

namespace paddle {
namespace lite_api {

/**
 * Not sure why, we have to initial a pointer first for kernels.
 * Otherwise it throws null pointer error when do KernelRegistor.
 */
static void use_arm_kernels() {
  ARM_KERNEL_POINTER(BatchNormCompute);
  ARM_KERNEL_POINTER(CalibComputeFp32ToInt8);
  ARM_KERNEL_POINTER(CalibComputeInt8ToFp32);
  ARM_KERNEL_POINTER(ConvCompute);
  ARM_KERNEL_POINTER(ConcatCompute);
  ARM_KERNEL_POINTER(ElementwiseAddCompute);
  ARM_KERNEL_POINTER(DropoutCompute);
  ARM_KERNEL_POINTER(FcCompute);
  ARM_KERNEL_POINTER(MulCompute);
  ARM_KERNEL_POINTER(PoolCompute);
  ARM_KERNEL_POINTER(ReluCompute);
  ARM_KERNEL_POINTER(ScaleCompute);
  ARM_KERNEL_POINTER(SoftmaxCompute);
  ARM_KERNEL_POINTER(SplitCompute);
  ARM_KERNEL_POINTER(TransposeCompute);
  ARM_KERNEL_POINTER(Transpose2Compute);
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_lite_PaddleLiteInitializer_initNative(JNIEnv *env,
                                                            jclass jclazz) {
  use_arm_kernels();
}

}  // namespace lite_api
}  // namespace paddle

#ifdef __cplusplus
}
#endif
