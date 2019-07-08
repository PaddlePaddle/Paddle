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

#include "paddle/fluid/lite/api/android/jni/native/paddle_lite_jni.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/lite/api/android/jni/native/convert_util_jni.h"
#include "paddle/fluid/lite/api/light_api.h"
#include "paddle/fluid/lite/api/paddle_api.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace paddle {
namespace lite_api {

inline static std::shared_ptr<PaddlePredictor> *getPaddlePredictorPointer(
    JNIEnv *env, jobject jpaddle_predictor) {
  jclass jclazz = env->GetObjectClass(jpaddle_predictor);
  jfieldID jfield = env->GetFieldID(jclazz, "cppPaddlePredictorPointer", "J");
  jlong java_pointer = env->GetLongField(jpaddle_predictor, jfield);
  std::shared_ptr<PaddlePredictor> *ptr =
      reinterpret_cast<std::shared_ptr<PaddlePredictor> *>(java_pointer);
  return ptr;
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_lite_PaddlePredictor_run(
    JNIEnv *env, jobject jpaddle_predictor) {
  std::shared_ptr<PaddlePredictor> *predictor =
      getPaddlePredictorPointer(env, jpaddle_predictor);
  if (predictor == nullptr || (*predictor == nullptr)) {
    return JNI_FALSE;
  }
  (*predictor)->Run();
  return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_saveOptimizedModel(
    JNIEnv *env, jobject jpaddle_predictor, jstring model_dir) {
  std::shared_ptr<PaddlePredictor> *predictor =
      getPaddlePredictorPointer(env, jpaddle_predictor);
  if (predictor == nullptr || (*predictor == nullptr)) {
    return JNI_FALSE;
  }
  (*predictor)->SaveOptimizedModel(jstring_to_cpp_string(env, model_dir));
  return JNI_TRUE;
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_getInputCppTensorPointer(
    JNIEnv *env, jobject jpaddle_predictor, jint offset) {
  std::shared_ptr<PaddlePredictor> *predictor =
      getPaddlePredictorPointer(env, jpaddle_predictor);
  if (predictor == nullptr || (*predictor == nullptr)) {
    return 0;
  }
  std::unique_ptr<Tensor> tensor =
      (*predictor)->GetInput(static_cast<int>(offset));
  std::unique_ptr<Tensor> *cpp_tensor_pointer =
      new std::unique_ptr<Tensor>(std::move(tensor));
  return reinterpret_cast<jlong>(cpp_tensor_pointer);
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_getOutputCppTensorPointer(
    JNIEnv *env, jobject jpaddle_predictor, jint offset) {
  std::shared_ptr<PaddlePredictor> *predictor =
      getPaddlePredictorPointer(env, jpaddle_predictor);
  if (predictor == nullptr || (*predictor == nullptr)) {
    return 0;
  }
  std::unique_ptr<const Tensor> tensor =
      (*predictor)->GetOutput(static_cast<int>(offset));
  std::unique_ptr<const Tensor> *cpp_tensor_pointer =
      new std::unique_ptr<const Tensor>(std::move(tensor));
  return reinterpret_cast<jlong>(cpp_tensor_pointer);
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_getCppTensorPointerByName(
    JNIEnv *env, jobject jpaddle_predictor, jstring name) {
  std::string cpp_name = jstring_to_cpp_string(env, name);
  std::shared_ptr<PaddlePredictor> *predictor =
      getPaddlePredictorPointer(env, jpaddle_predictor);
  if (predictor == nullptr || (*predictor == nullptr)) {
    return 0;
  }
  std::unique_ptr<const Tensor> tensor = (*predictor)->GetTensor(cpp_name);
  std::unique_ptr<const Tensor> *cpp_tensor_pointer =
      new std::unique_ptr<const Tensor>(std::move(tensor));
  return reinterpret_cast<jlong>(cpp_tensor_pointer);
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_newCppPaddlePredictor__Lcom_baidu_\
paddle_lite_CxxConfig_2(JNIEnv *env, jobject jpaddle_predictor,
                        jobject jcxxconfig) {
  CxxConfig config = jcxxconfig_to_cpp_cxxconfig(env, jcxxconfig);
  std::shared_ptr<PaddlePredictor> predictor =
      paddle::lite_api::CreatePaddlePredictor(config);
  if (predictor == nullptr) {
    return 0;
  }
  std::shared_ptr<PaddlePredictor> *predictor_pointer =
      new std::shared_ptr<PaddlePredictor>(predictor);
  return reinterpret_cast<jlong>(predictor_pointer);
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_newCppPaddlePredictor__Lcom_baidu_\
paddle_lite_MobileConfig_2(JNIEnv *env, jobject jpaddle_predictor,
                           jobject jmobileconfig) {
  MobileConfig config = jmobileconfig_to_cpp_mobileconfig(env, jmobileconfig);
  std::shared_ptr<PaddlePredictor> predictor =
      paddle::lite_api::CreatePaddlePredictor(config);
  if (predictor == nullptr) {
    return 0;
  }
  std::shared_ptr<PaddlePredictor> *predictor_pointer =
      new std::shared_ptr<PaddlePredictor>(predictor);
  return reinterpret_cast<jlong>(predictor_pointer);
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_PaddlePredictor_deleteCppPaddlePredictor(
    JNIEnv *env, jobject jpaddle_predictor, jlong java_pointer) {
  if (java_pointer == 0) {
    return JNI_FALSE;
  }
  std::shared_ptr<PaddlePredictor> *ptr =
      reinterpret_cast<std::shared_ptr<PaddlePredictor> *>(java_pointer);
  ptr->reset();
  delete ptr;
  return JNI_TRUE;
}

}  // namespace lite_api
}  // namespace paddle

#ifdef __cplusplus
}
#endif
