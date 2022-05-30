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

#include "com_baidu_paddle_inference_Tensor.h"
#include <jni.h>
#include "pd_inference_api.h"  // NOLINT

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_cppTensorDestroy(
    JNIEnv *, jobject, jlong tensorPointer) {
  PD_TensorDestroy(reinterpret_cast<PD_Tensor *>(tensorPointer));
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_cppTensorReshape(
    JNIEnv *env, jobject, jlong tensorPointer, jint dim, jintArray array) {
  int32_t *input_shape = env->GetIntArrayElements(array, nullptr);
  PD_TensorReshape(reinterpret_cast<PD_Tensor *>(tensorPointer),
                   static_cast<int>(dim), input_shape);
  env->ReleaseIntArrayElements(array, input_shape, JNI_ABORT);
}

JNIEXPORT jintArray JNICALL
Java_com_baidu_paddle_inference_Tensor_cppTensorGetShape(JNIEnv *env, jobject,
                                                         jlong tensorPointer) {
  PD_Tensor *tensor = reinterpret_cast<PD_Tensor *>(tensorPointer);
  PD_OneDimArrayInt32 *output_shape = PD_TensorGetShape(tensor);
  jintArray result = env->NewIntArray(output_shape->size);
  env->SetIntArrayRegion(result, 0, output_shape->size, output_shape->data);
  return result;
}

JNIEXPORT jstring JNICALL
Java_com_baidu_paddle_inference_Tensor_cppTensorGetName(JNIEnv *env, jobject,
                                                        jlong tensorPointer) {
  const char *c_str =
      PD_TensorGetName(reinterpret_cast<PD_Tensor *>(tensorPointer));
  return env->NewStringUTF(c_str);
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_cppTensorCopyFromCpuFloat(
    JNIEnv *env, jobject, jlong tensorPointer, jfloatArray array) {
  float *data = env->GetFloatArrayElements(array, nullptr);
  PD_TensorCopyFromCpuFloat(reinterpret_cast<PD_Tensor *>(tensorPointer), data);
  env->ReleaseFloatArrayElements(array, data, JNI_ABORT);
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_cppTensorCopyFromCpuInt(
    JNIEnv *env, jobject, jlong tensorPointer, jintArray array) {
  int32_t *data = env->GetIntArrayElements(array, nullptr);
  PD_TensorCopyFromCpuInt32(reinterpret_cast<PD_Tensor *>(tensorPointer), data);
  env->ReleaseIntArrayElements(array, data, 0);
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_cppTensorCopyFromCpuLong(
    JNIEnv *env, jobject, jlong tensorPointer, jlongArray array) {
  int64_t *data = env->GetLongArrayElements(array, nullptr);
  PD_TensorCopyFromCpuInt64(reinterpret_cast<PD_Tensor *>(tensorPointer), data);
  env->ReleaseLongArrayElements(array, data, 0);
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_cppTensorCopyFromCpuByte(
    JNIEnv *env, jobject, jlong tensorPointer, jbyteArray array) {
  int8_t *data = env->GetByteArrayElements(array, nullptr);
  PD_TensorCopyFromCpuInt8(reinterpret_cast<PD_Tensor *>(tensorPointer), data);
  env->ReleaseByteArrayElements(array, data, 0);
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_cppTensorCopyFromCpuBoolean(
    JNIEnv *env, jobject, jlong tensorPointer, jbooleanArray array) {
  uint8_t *data = env->GetBooleanArrayElements(array, nullptr);
  PD_TensorCopyFromCpuUint8(reinterpret_cast<PD_Tensor *>(tensorPointer), data);
  env->ReleaseBooleanArrayElements(array, data, 0);
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_cppTensorCopyToCpuFloat(
    JNIEnv *env, jobject, jlong tensorPointer, jfloatArray array) {
  float *data = env->GetFloatArrayElements(array, nullptr);
  PD_TensorCopyToCpuFloat(reinterpret_cast<PD_Tensor *>(tensorPointer), data);
  env->ReleaseFloatArrayElements(array, data, 0);
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_cppTensorCopyToCpuInt(
    JNIEnv *env, jobject, jlong tensorPointer, jintArray array) {
  int32_t *data = env->GetIntArrayElements(array, nullptr);
  PD_TensorCopyToCpuInt32(reinterpret_cast<PD_Tensor *>(tensorPointer), data);
  env->ReleaseIntArrayElements(array, data, 0);
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_cppTensorCopyToCpuLong(
    JNIEnv *env, jobject, jlong tensorPointer, jlongArray array) {
  int64_t *data = env->GetLongArrayElements(array, nullptr);
  PD_TensorCopyToCpuInt64(reinterpret_cast<PD_Tensor *>(tensorPointer), data);
  env->ReleaseLongArrayElements(array, data, 0);
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_cppTensorCopyToCpuByte(
    JNIEnv *env, jobject, jlong tensorPointer, jbyteArray array) {
  int8_t *data = env->GetByteArrayElements(array, nullptr);
  PD_TensorCopyToCpuInt8(reinterpret_cast<PD_Tensor *>(tensorPointer), data);
  env->ReleaseByteArrayElements(array, data, 0);
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_cppTensorCopyToCpuBoolean(
    JNIEnv *env, jobject, jlong tensorPointer, jbooleanArray array) {
  uint8_t *data = env->GetBooleanArrayElements(array, nullptr);
  PD_TensorCopyToCpuUint8(reinterpret_cast<PD_Tensor *>(tensorPointer), data);
  env->ReleaseBooleanArrayElements(array, data, 0);
}
