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

#include <jni.h>
#include <string>
#include <vector>

#include "paddle/fluid/lite/api/light_api.h"
#include "paddle/fluid/lite/api/paddle_api.h"
#include "paddle/fluid/lite/api/paddle_place.h"

#ifndef PADDLE_FLUID_LITE_API_ANDROID_JNI_NATIVE_CONVERT_UTIL_JNI_H_
#define PADDLE_FLUID_LITE_API_ANDROID_JNI_NATIVE_CONVERT_UTIL_JNI_H_

namespace paddle {
namespace lite_api {

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

/**
 * Converts Java com.baidu.paddle.lite.Place to c++ paddle::lite_api::Place.
 */
inline Place jplace_to_cpp_place(JNIEnv *env, jobject java_place) {
  jclass place_jclazz = env->GetObjectClass(java_place);

  jmethodID target_method =
      env->GetMethodID(place_jclazz, "getTargetInt", "()I");
  jmethodID precision_method =
      env->GetMethodID(place_jclazz, "getPrecisionInt", "()I");
  jmethodID data_layout_method =
      env->GetMethodID(place_jclazz, "getDataLayoutInt", "()I");
  jmethodID device_method = env->GetMethodID(place_jclazz, "getDevice", "()I");

  int target = env->CallIntMethod(java_place, target_method);
  int precision = env->CallIntMethod(java_place, precision_method);
  int data_layout = env->CallIntMethod(java_place, data_layout_method);
  int device = env->CallIntMethod(java_place, device_method);

  return Place(static_cast<paddle::lite_api::TargetType>(target),
               static_cast<paddle::lite_api::PrecisionType>(precision),
               static_cast<paddle::lite_api::DataLayoutType>(data_layout),
               device);
}

inline CxxConfig jcxxconfig_to_cpp_cxxconfig(JNIEnv *env, jobject jcxxconfig) {
  jclass cxxconfig_jclazz = env->GetObjectClass(jcxxconfig);

  jmethodID model_dir_method =
      env->GetMethodID(cxxconfig_jclazz, "getModelDir", "()Ljava/lang/String;");
  jmethodID preferred_place_method = env->GetMethodID(
      cxxconfig_jclazz, "getPreferredPlace", "()Lcom/baidu/paddle/lite/Place;");
  jmethodID valid_places_method = env->GetMethodID(
      cxxconfig_jclazz, "getValidPlaces", "()[Lcom/baidu/paddle/lite/Place;");

  CxxConfig config;

  jstring java_model_dir =
      (jstring)env->CallObjectMethod(jcxxconfig, model_dir_method);
  if (java_model_dir != nullptr) {
    std::string cpp_model_dir = jstring_to_cpp_string(env, java_model_dir);
    config.set_model_dir(cpp_model_dir);
  }

  jobject java_preferred_place =
      env->CallObjectMethod(jcxxconfig, preferred_place_method);
  if (java_preferred_place != nullptr) {
    Place cpp_preferred_place = jplace_to_cpp_place(env, java_preferred_place);
    config.set_preferred_place(cpp_preferred_place);
  }

  jobject object_valid_places =
      env->CallObjectMethod(jcxxconfig, valid_places_method);
  jobjectArray *java_valid_places =
      reinterpret_cast<jobjectArray *>(&object_valid_places);
  if (java_valid_places != nullptr) {
    int valid_place_count = env->GetArrayLength(*java_valid_places);
    std::vector<Place> cpp_valid_places;
    for (int i = 0; i < valid_place_count; ++i) {
      jobject jplace = env->GetObjectArrayElement(*java_valid_places, i);
      cpp_valid_places.push_back(jplace_to_cpp_place(env, jplace));
    }
    config.set_valid_places(cpp_valid_places);
  }

  return config;
}

inline MobileConfig jmobileconfig_to_cpp_mobileconfig(JNIEnv *env,
                                                      jobject jmobileconfig) {
  jclass mobileconfig_jclazz = env->GetObjectClass(jmobileconfig);

  jmethodID model_dir_method = env->GetMethodID(
      mobileconfig_jclazz, "getModelDir", "()Ljava/lang/String;");
  MobileConfig config;

  jstring java_model_dir =
      (jstring)env->CallObjectMethod(jmobileconfig, model_dir_method);
  if (java_model_dir != nullptr) {
    std::string cpp_model_dir = jstring_to_cpp_string(env, java_model_dir);
    config.set_model_dir(cpp_model_dir);
  }
  return config;
}

}  // namespace lite_api
}  // namespace paddle

#endif  //  PADDLE_FLUID_LITE_API_ANDROID_JNI_NATIVE_CONVERT_UTIL_JNI_H_
