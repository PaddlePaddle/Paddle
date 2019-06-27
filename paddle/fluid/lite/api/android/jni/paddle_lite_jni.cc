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

#include "paddle/fluid/lite/api/android/jni/paddle_lite_jni.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/lite/kernels/arm/activation_compute.h"
#include "paddle/fluid/lite/kernels/arm/batch_norm_compute.h"
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

#include "paddle/fluid/lite/api/light_api.h"
#include "paddle/fluid/lite/api/paddle_api.h"
#include "paddle/fluid/lite/api/paddle_lite_factory_helper.h"
#include "paddle/fluid/lite/api/paddle_use_kernels.h"
#include "paddle/fluid/lite/api/paddle_use_ops.h"
#include "paddle/fluid/lite/api/paddle_use_passes.h"

#define ARM_KERNEL_POINTER(kernel_class_name__)                    \
  std::unique_ptr<paddle::lite::kernels::arm::kernel_class_name__> \
      p##kernel_class_name__(                                      \
          new paddle::lite::kernels::arm::kernel_class_name__);

#ifdef __cplusplus
extern "C" {
#endif

using paddle::lite_api::MobileConfig;
using paddle::lite_api::PaddlePredictor;
using paddle::lite_api::Tensor;

static std::shared_ptr<PaddlePredictor> predictor;

/**
 * Not sure why, we have to initial a pointer first for kernels.
 * Otherwise it throws null pointer error when do KernelRegistor.
 */
static void use_arm_kernels() {
  ARM_KERNEL_POINTER(BatchNormCompute);
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

inline std::string jstring_to_cpp_string(JNIEnv *env, jstring jstr) {
  // In java, a unicode char will be encoded using 2 bytes (utf16).
  // so jstring will contain characters utf16. std::string in c++ is
  // essentially a string of bytes, not characters, so if we want to
  // pass jstring from JNI to c++, we have convert utf16 to bytes.
  if (!jstr) {
    return "";
  }
  const jclass stringClass = env->GetObjectClass(jstr);
  const jmethodID getBytes =
      env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
  const jbyteArray stringJbytes = (jbyteArray)env->CallObjectMethod(
      jstr, getBytes, env->NewStringUTF("UTF-8"));

  size_t length = (size_t)env->GetArrayLength(stringJbytes);
  jbyte *pBytes = env->GetByteArrayElements(stringJbytes, NULL);

  std::string ret = std::string(reinterpret_cast<char *>(pBytes), length);
  env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

  env->DeleteLocalRef(stringJbytes);
  env->DeleteLocalRef(stringClass);
  return ret;
}

inline jfloatArray cpp_array_to_jfloatarray(JNIEnv *env, const float *buf,
                                            int64_t len) {
  jfloatArray result = env->NewFloatArray(len);
  env->SetFloatArrayRegion(result, 0, len, buf);
  return result;
}

inline jintArray cpp_array_to_jintarray(JNIEnv *env, const int *buf,
                                        int64_t len) {
  jintArray result = env->NewIntArray(len);
  env->SetIntArrayRegion(result, 0, len, buf);
  return result;
}

inline jbyteArray cpp_array_to_jbytearray(JNIEnv *env, const int8_t *buf,
                                          int64_t len) {
  jbyteArray result = env->NewByteArray(len);
  env->SetByteArrayRegion(result, 0, len, buf);
  return result;
}

inline std::vector<int64_t> jintarray_to_int64_vector(JNIEnv *env,
                                                      jintArray dims) {
  int dim_size = env->GetArrayLength(dims);
  jint *dim_nums = env->GetIntArrayElements(dims, nullptr);
  std::vector<int64_t> dim_vec(dim_nums, dim_nums + dim_size);
  env->ReleaseIntArrayElements(dims, dim_nums, 0);
  return dim_vec;
}

inline static int64_t product(const std::vector<int64_t> &vec) {
  if (vec.empty()) {
    return 0;
  }
  int64_t result = 1;
  for (int64_t d : vec) {
    result *= d;
  }
  return result;
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_loadMobileModel(JNIEnv *env,
                                                           jclass thiz,
                                                           jstring model_path) {
  if (predictor != nullptr) {
    return JNI_FALSE;
  }
  use_arm_kernels();
  MobileConfig config;
  std::string model_dir = jstring_to_cpp_string(env, model_path);
  config.set_model_dir(model_dir);
  predictor = paddle::lite_api::CreatePaddlePredictor(config);
  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_clear(JNIEnv *env, jclass thiz) {
  if (predictor == nullptr) {
    return JNI_FALSE;
  }
  predictor.reset();
  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_setInput__I_3I_3F(
    JNIEnv *env, jclass thiz, jint offset, jintArray dims, jfloatArray buf) {
  std::vector<int64_t> ddim = jintarray_to_int64_vector(env, dims);

  int len = env->GetArrayLength(buf);
  if ((int64_t)len != product(ddim)) {
    return JNI_FALSE;
  }

  float *buffer = env->GetFloatArrayElements(buf, nullptr);
  std::unique_ptr<Tensor> tensor =
      predictor->GetInput(static_cast<int>(offset));
  tensor->Resize(ddim);
  float *input = tensor->mutable_data<float>();
  for (int i = 0; i < len; ++i) {
    input[i] = buffer[i];
  }
  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_setInput__I_3I_3B(
    JNIEnv *env, jclass thiz, jint offset, jintArray dims, jbyteArray buf) {
  std::vector<int64_t> ddim = jintarray_to_int64_vector(env, dims);

  int len = env->GetArrayLength(buf);
  if ((int64_t)len != product(ddim)) {
    return JNI_FALSE;
  }

  jbyte *buffer = env->GetByteArrayElements(buf, nullptr);
  std::unique_ptr<Tensor> tensor =
      predictor->GetInput(static_cast<int>(offset));
  tensor->Resize(ddim);
  int8_t *input = tensor->mutable_data<int8_t>();
  for (int i = 0; i < len; ++i) {
    input[i] = (int8_t)buffer[i];
  }

  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_run(JNIEnv *, jclass) {
  predictor->Run();
  return JNI_TRUE;
}

JNIEXPORT jfloatArray JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_getFloatOutput(JNIEnv *env,
                                                          jclass thiz,
                                                          jint offset) {
  std::unique_ptr<const Tensor> tensor =
      predictor->GetOutput(static_cast<int>(offset));
  int64_t len = product(tensor->shape());
  return cpp_array_to_jfloatarray(env, tensor->data<float>(), len);
}

JNIEXPORT jbyteArray JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_getByteOutput(JNIEnv *env,
                                                         jclass thiz,
                                                         jint offset) {
  std::unique_ptr<const Tensor> tensor =
      predictor->GetOutput(static_cast<int>(offset));
  int64_t len = product(tensor->shape());
  return cpp_array_to_jbytearray(env, tensor->data<int8_t>(), len);
}

JNIEXPORT jfloatArray JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_fetchFloat(JNIEnv *env, jclass thiz,
                                                      jstring name) {
  std::string cpp_name = jstring_to_cpp_string(env, name);
  std::unique_ptr<const Tensor> tensor = predictor->GetTensor(cpp_name);
  int64_t len = product(tensor->shape());
  return cpp_array_to_jfloatarray(env, tensor->data<float>(), len);
}

JNIEXPORT jbyteArray JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_fetchByte(JNIEnv *env, jclass thiz,
                                                     jstring name) {
  std::string cpp_name = jstring_to_cpp_string(env, name);
  std::unique_ptr<const Tensor> tensor = predictor->GetTensor(cpp_name);
  int64_t len = product(tensor->shape());
  return cpp_array_to_jbytearray(env, tensor->data<int8_t>(), len);
}

#ifdef __cplusplus
}
#endif
