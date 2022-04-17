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

#ifndef PADDLE_FLUID_INFERENCE_JAVAAPI_NATIVE_JNI_CONVERT_UTIL_H_  // NOLINT
#define PADDLE_FLUID_INFERENCE_JAVAAPI_NATIVE_JNI_CONVERT_UTIL_H_

#include <jni.h>
#include <string.h>
#include <string>
#include <vector>

#define PADDLE_WITH_CUDA PADDLE_WITH_CUDA

inline std::string jstring_to_cpp_string(JNIEnv *env, jstring jstr) {
  if (!jstr) {
    return "";
  }
  const jclass stringClass = env->GetObjectClass(jstr);
  const jmethodID getBytes =
      env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
  const jbyteArray stringJbytes = (jbyteArray)env->CallObjectMethod(
      jstr, getBytes, env->NewStringUTF("UTF-8"));

  size_t length = static_cast<size_t>(env->GetArrayLength(stringJbytes));
  jbyte *pBytes = env->GetByteArrayElements(stringJbytes, NULL);

  std::string ret = std::string(reinterpret_cast<char *>(pBytes), length);
  env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

  env->DeleteLocalRef(stringJbytes);
  env->DeleteLocalRef(stringClass);
  return ret;
}

inline jstring cpp_string_to_jstring(JNIEnv *env, std::string str) {
  auto *data = str.c_str();
  jclass strClass = env->FindClass("java/lang/String");
  jmethodID strClassInitMethodID =
      env->GetMethodID(strClass, "<init>", "([BLjava/lang/String;)V");

  jbyteArray bytes = env->NewByteArray(strlen(data));
  env->SetByteArrayRegion(bytes, 0, strlen(data),
                          reinterpret_cast<const jbyte *>(data));

  jstring encoding = env->NewStringUTF("UTF-8");
  jstring res = (jstring)(
      env->NewObject(strClass, strClassInitMethodID, bytes, encoding));

  env->DeleteLocalRef(strClass);
  env->DeleteLocalRef(encoding);
  env->DeleteLocalRef(bytes);

  return res;
}

inline jboolean cpp_bool_to_jboolean(JNIEnv *env, bool flag) {
  return flag ? JNI_TRUE : JNI_FALSE;
}

inline bool jboolean_to_cpp_bool(JNIEnv *env, jboolean flag) {
  return flag == JNI_TRUE;
}

inline jfloatArray cpp_array_to_jfloatarray(JNIEnv *env, const float *buf,
                                            int64_t len) {
  jfloatArray result = env->NewFloatArray(len);
  env->SetFloatArrayRegion(result, 0, len, buf);
  return result;
}

inline jbyteArray cpp_array_to_jbytearray(JNIEnv *env, const int8_t *buf,
                                          int64_t len) {
  jbyteArray result = env->NewByteArray(len);
  env->SetByteArrayRegion(result, 0, len, buf);
  return result;
}

inline jintArray cpp_array_to_jintarray(JNIEnv *env, const int *buf,
                                        int64_t len) {
  jintArray result = env->NewIntArray(len);
  env->SetIntArrayRegion(result, 0, len, buf);
  return result;
}

inline jlongArray cpp_array_to_jlongarray(JNIEnv *env, const int64_t *buf,
                                          int64_t len) {
  jlongArray result = env->NewLongArray(len);
  env->SetLongArrayRegion(result, 0, len, buf);
  return result;
}

inline jlongArray int64_vector_to_jlongarray(JNIEnv *env,
                                             const std::vector<int64_t> &vec) {
  jlongArray result = env->NewLongArray(vec.size());
  jlong *buf = new jlong[vec.size()];
  for (size_t i = 0; i < vec.size(); ++i) {
    buf[i] = (jlong)vec[i];
  }
  env->SetLongArrayRegion(result, 0, vec.size(), buf);
  delete[] buf;
  return result;
}

inline std::vector<int64_t> jlongarray_to_int64_vector(JNIEnv *env,
                                                       jlongArray dims) {
  int dim_size = env->GetArrayLength(dims);
  jlong *dim_nums = env->GetLongArrayElements(dims, nullptr);
  std::vector<int64_t> dim_vec(dim_nums, dim_nums + dim_size);
  env->ReleaseLongArrayElements(dims, dim_nums, 0);
  return dim_vec;
}
#endif  // PADDLE_FLUID_INFERENCE_JAVAAPI_NATIVE_JNI_CONVERT_UTIL_H_
