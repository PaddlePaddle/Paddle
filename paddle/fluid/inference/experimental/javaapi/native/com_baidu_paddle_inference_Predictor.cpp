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

#include "com_baidu_paddle_inference_Predictor.h"
#include <jni.h>
#include "jni_convert_util.h"  // NOLINT
#include "pd_inference_api.h"  // NOLINT

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Predictor_cppPredictorDestroy(
    JNIEnv*, jobject, jlong cppPaddlePredictorPointer) {
  PD_PredictorDestroy(
      reinterpret_cast<PD_Predictor*>(cppPaddlePredictorPointer));
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Predictor_predictorTryShrinkMemory(
    JNIEnv* env, jobject obj, jlong cppPaddlePredictorPointer) {
  PD_PredictorTryShrinkMemory(
      reinterpret_cast<PD_Predictor*>(cppPaddlePredictorPointer));
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Predictor_predictorClearIntermediateTensor(
    JNIEnv* env, jobject obj, jlong cppPaddlePredictorPointer) {
  PD_PredictorClearIntermediateTensor(
      reinterpret_cast<PD_Predictor*>(cppPaddlePredictorPointer));
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_inference_Predictor_createPredictor(
    JNIEnv* env, jobject obj, jlong cppPaddlePredictorPointer) {
  return (jlong)PD_PredictorCreate(
      reinterpret_cast<PD_Config*>(cppPaddlePredictorPointer));
}

JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Predictor_getInputNum(
    JNIEnv* env, jobject obj, jlong cppPaddlePredictorPointer) {
  return (jlong)PD_PredictorGetInputNum(
      reinterpret_cast<PD_Predictor*>(cppPaddlePredictorPointer));
}

JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Predictor_getOutputNum(
    JNIEnv* env, jobject obj, jlong cppPaddlePredictorPointer) {
  return (jlong)PD_PredictorGetOutputNum(
      reinterpret_cast<PD_Predictor*>(cppPaddlePredictorPointer));
}

JNIEXPORT jstring JNICALL
Java_com_baidu_paddle_inference_Predictor_getInputNameByIndex(
    JNIEnv* env, jobject obj, jlong cppPaddlePredictorPointer, jlong index) {
  const char* c_str = PD_PredictorGetInputNames(reinterpret_cast<PD_Predictor*>(
                                                    cppPaddlePredictorPointer))
                          ->data[static_cast<int>(index)];
  return env->NewStringUTF(c_str);
}

JNIEXPORT jstring JNICALL
Java_com_baidu_paddle_inference_Predictor_getOutputNameByIndex(
    JNIEnv* env, jobject obj, jlong cppPaddlePredictorPointer, jlong index) {
  const char* c_str =
      PD_PredictorGetOutputNames(
          reinterpret_cast<PD_Predictor*>(cppPaddlePredictorPointer))
          ->data[static_cast<int>(index)];
  return env->NewStringUTF(c_str);
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_inference_Predictor_getInputHandleByName(
    JNIEnv* env, jobject obj, jlong cppPaddlePredictorPointer, jstring name) {
  // const char* input_name = env->GetStringUTFChars(name, 0);
  PD_Predictor* pd_predictor =
      reinterpret_cast<PD_Predictor*>(cppPaddlePredictorPointer);
  jlong output_tensor = (jlong)PD_PredictorGetInputHandle(
      pd_predictor, jstring_to_cpp_string(env, name).c_str());
  return output_tensor;
}

JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_inference_Predictor_getOutputHandleByName(
    JNIEnv* env, jobject obj, jlong cppPaddlePredictorPointer, jstring name) {
  // const char* output_name = env->GetStringUTFChars(name, 0);
  PD_Predictor* pd_predictor =
      reinterpret_cast<PD_Predictor*>(cppPaddlePredictorPointer);
  jlong output_tensor = (jlong)PD_PredictorGetOutputHandle(
      pd_predictor, jstring_to_cpp_string(env, name).c_str());
  return output_tensor;
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_inference_Predictor_runPD(
    JNIEnv* env, jobject obj, jlong cppPaddlePredictorPointer) {
  return (jboolean)PD_PredictorRun(
      reinterpret_cast<PD_Predictor*>(cppPaddlePredictorPointer));
}
