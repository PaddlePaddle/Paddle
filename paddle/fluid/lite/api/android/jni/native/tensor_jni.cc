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

#include "paddle/fluid/lite/api/android/jni/native/tensor_jni.h"

#include <memory>
#include <vector>

#include "paddle/fluid/lite/api/android/jni/native/convert_util_jni.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace paddle {
namespace lite_api {

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

inline static bool is_const_tensor(JNIEnv *env, jobject jtensor) {
  jclass jclazz = env->GetObjectClass(jtensor);
  jfieldID jfield = env->GetFieldID(jclazz, "readOnly", "Z");
  jboolean read_only = env->GetBooleanField(jtensor, jfield);
  return static_cast<bool>(read_only);
}

inline static std::unique_ptr<Tensor> *get_writable_tensor_pointer(
    JNIEnv *env, jobject jtensor) {
  jclass jclazz = env->GetObjectClass(jtensor);
  jfieldID jfield = env->GetFieldID(jclazz, "cppTensorPointer", "J");
  jlong java_pointer = env->GetLongField(jtensor, jfield);
  std::unique_ptr<Tensor> *ptr =
      reinterpret_cast<std::unique_ptr<Tensor> *>(java_pointer);
  return ptr;
}

inline static std::unique_ptr<const Tensor> *get_read_only_tensor_pointer(
    JNIEnv *env, jobject jtensor) {
  jclass jclazz = env->GetObjectClass(jtensor);
  jfieldID jfield = env->GetFieldID(jclazz, "cppTensorPointer", "J");
  jlong java_pointer = env->GetLongField(jtensor, jfield);
  std::unique_ptr<const Tensor> *ptr =
      reinterpret_cast<std::unique_ptr<const Tensor> *>(java_pointer);
  return ptr;
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_lite_Tensor_resize(
    JNIEnv *env, jobject jtensor, jlongArray dims) {
  std::unique_ptr<Tensor> *tensor = get_writable_tensor_pointer(env, jtensor);
  if (tensor == nullptr || (*tensor == nullptr)) {
    return JNI_FALSE;
  }
  std::vector<int64_t> shape = jlongarray_to_int64_vector(env, dims);
  (*tensor)->Resize(shape);
  return JNI_TRUE;
}

JNIEXPORT jlongArray JNICALL
Java_com_baidu_paddle_lite_Tensor_shape(JNIEnv *env, jobject jtensor) {
  if (is_const_tensor(env, jtensor)) {
    std::unique_ptr<const Tensor> *tensor =
        get_read_only_tensor_pointer(env, jtensor);
    std::vector<int64_t> shape = (*tensor)->shape();
    return int64_vector_to_jlongarray(env, shape);
  } else {
    std::unique_ptr<Tensor> *tensor = get_writable_tensor_pointer(env, jtensor);
    std::vector<int64_t> shape = (*tensor)->shape();
    return int64_vector_to_jlongarray(env, shape);
  }
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_lite_Tensor_setData___3F(
    JNIEnv *env, jobject jtensor, jfloatArray buf) {
  std::unique_ptr<Tensor> *tensor = get_writable_tensor_pointer(env, jtensor);
  if (tensor == nullptr || (*tensor == nullptr)) {
    return JNI_FALSE;
  }
  int64_t buf_size = (int64_t)env->GetArrayLength(buf);
  if (buf_size != product((*tensor)->shape())) {
    return JNI_FALSE;
  }

  float *input = (*tensor)->mutable_data<float>();
  env->GetFloatArrayRegion(buf, 0, buf_size, input);
  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_lite_Tensor_setData___3B(
    JNIEnv *env, jobject jtensor, jbyteArray buf) {
  std::unique_ptr<Tensor> *tensor = get_writable_tensor_pointer(env, jtensor);
  if (tensor == nullptr || (*tensor == nullptr)) {
    return JNI_FALSE;
  }
  int64_t buf_size = (int64_t)env->GetArrayLength(buf);
  if (buf_size != product((*tensor)->shape())) {
    return JNI_FALSE;
  }

  int8_t *input = (*tensor)->mutable_data<int8_t>();
  env->GetByteArrayRegion(buf, 0, buf_size, input);
  return JNI_TRUE;
}

JNIEXPORT jfloatArray JNICALL
Java_com_baidu_paddle_lite_Tensor_getFloatData(JNIEnv *env, jobject jtensor) {
  if (is_const_tensor(env, jtensor)) {
    std::unique_ptr<const Tensor> *tensor =
        get_read_only_tensor_pointer(env, jtensor);
    return cpp_array_to_jfloatarray(env, (*tensor)->data<float>(),
                                    product((*tensor)->shape()));
  } else {
    std::unique_ptr<Tensor> *tensor = get_writable_tensor_pointer(env, jtensor);
    return cpp_array_to_jfloatarray(env, (*tensor)->data<float>(),
                                    product((*tensor)->shape()));
  }
}

JNIEXPORT jbyteArray JNICALL
Java_com_baidu_paddle_lite_Tensor_getByteData(JNIEnv *env, jobject jtensor) {
  if (is_const_tensor(env, jtensor)) {
    std::unique_ptr<const Tensor> *tensor =
        get_read_only_tensor_pointer(env, jtensor);
    return cpp_array_to_jbytearray(env, (*tensor)->data<int8_t>(),
                                   product((*tensor)->shape()));
  } else {
    std::unique_ptr<Tensor> *tensor = get_writable_tensor_pointer(env, jtensor);
    return cpp_array_to_jbytearray(env, (*tensor)->data<int8_t>(),
                                   product((*tensor)->shape()));
  }
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_lite_Tensor_deleteCppTensor(
    JNIEnv *env, jobject jtensor, jlong java_pointer) {
  if (java_pointer == 0) {
    return JNI_FALSE;
  }
  std::unique_ptr<Tensor> *ptr =
      reinterpret_cast<std::unique_ptr<Tensor> *>(java_pointer);
  ptr->reset();
  delete ptr;
  return JNI_TRUE;
}

}  // namespace lite_api
}  // namespace paddle

#ifdef __cplusplus
}
#endif
