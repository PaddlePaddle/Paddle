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

#include "com_baidu_paddle_inference_Config.h"
#include <iostream>
#include "jni_convert_util.h"  // NOLINT

#include "pd_inference_api.h"  // NOLINT

// 1. create Config

JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Config_createCppConfig(
    JNIEnv* env, jobject obj) {
  jlong cppPaddleConfigPointer = (jlong)PD_ConfigCreate();
  return cppPaddleConfigPointer;
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_inference_Config_isCppConfigValid(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  bool flag =
      PD_ConfigIsValid(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  return cpp_bool_to_jboolean(env, flag);
}

// 2. not combined model settings

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_setCppModel(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer, jstring modelFile,
    jstring paramsFile) {
  const char* model_file = env->GetStringUTFChars(modelFile, 0);
  const char* params_file = env->GetStringUTFChars(paramsFile, 0);
  PD_ConfigSetModel(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer),
                    model_file, params_file);
  free(const_cast<char*>(model_file));
  free(const_cast<char*>(params_file));
}

// 3. combined model settings

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_setCppModelDir(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer, jstring modelDir) {
  const char* model_dir = env->GetStringUTFChars(modelDir, 0);
  PD_ConfigSetModelDir(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer),
                       model_dir);
  free(const_cast<char*>(model_dir));
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_setCppProgFile(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer, jstring progFile) {
  const char* prog_file = env->GetStringUTFChars(progFile, 0);
  PD_ConfigSetProgFile(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer),
                       prog_file);
  free(const_cast<char*>(prog_file));
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_setCppParamsFile(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer,
    jstring paramsFile) {
  const char* params_file = env->GetStringUTFChars(paramsFile, 0);
  PD_ConfigSetParamsFile(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer),
                         params_file);
  free(const_cast<char*>(params_file));
}

JNIEXPORT jstring JNICALL Java_com_baidu_paddle_inference_Config_modelDir(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  const char* model_dir = PD_ConfigGetModelDir(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  jstring modelDir = env->NewStringUTF(model_dir);
  free(const_cast<char*>(model_dir));
  return modelDir;
}

JNIEXPORT jstring JNICALL Java_com_baidu_paddle_inference_Config_progFile(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  const char* prog_file = PD_ConfigGetProgFile(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  jstring progFile = env->NewStringUTF(prog_file);
  free(const_cast<char*>(prog_file));
  return progFile;
}

JNIEXPORT jstring JNICALL Java_com_baidu_paddle_inference_Config_paramsFile(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  const char* params_file = PD_ConfigGetProgFile(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  jstring paramsFile = env->NewStringUTF(params_file);
  free(const_cast<char*>(params_file));
  return paramsFile;
}

// 4. cpu settings

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Config_setCpuMathLibraryNumThreads(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer,
    jint mathThreadsNum) {
  int math_threads_num = static_cast<int>(mathThreadsNum);
  PD_ConfigSetCpuMathLibraryNumThreads(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer), math_threads_num);
}

JNIEXPORT jint JNICALL
Java_com_baidu_paddle_inference_Config_cpuMathLibraryNumThreads(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  jint mathThreadsNum = (jint)PD_ConfigGetCpuMathLibraryNumThreads(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
}

// 5. MKLDNN settings

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_enableMKLDNN(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  PD_ConfigEnableMKLDNN(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_inference_Config_mkldnnEnabled(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  bool flag = PD_ConfigMkldnnEnabled(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  return cpp_bool_to_jboolean(env, flag);
}

JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Config_enableMkldnnBfloat16(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  PD_ConfigEnableMkldnnBfloat16(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_inference_Config_mkldnnBfloat16Enabled(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  bool flag = PD_ConfigMkldnnBfloat16Enabled(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  return cpp_bool_to_jboolean(env, flag);
}

// 6. gpu setting

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_enableUseGpu(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer, jlong memorySize,
    jint deviceId) {
  PD_ConfigEnableUseGpu(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer),
                        (uint64_t)memorySize, (int32_t)deviceId);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_disableGpu(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  PD_ConfigDisableGpu(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_inference_Config_useGpu(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  bool flag =
      PD_ConfigUseGpu(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  return cpp_bool_to_jboolean(env, flag);
}

JNIEXPORT jint JNICALL Java_com_baidu_paddle_inference_Config_gpuDeviceId(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  int device_id = PD_ConfigGpuDeviceId(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  return (jint)device_id;
}

JNIEXPORT jint JNICALL
Java_com_baidu_paddle_inference_Config_memoryPoolInitSizeMb(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  int memory_pool_init_size_mb = PD_ConfigMemoryPoolInitSizeMb(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  return (jint)memory_pool_init_size_mb;
}

JNIEXPORT jfloat JNICALL
Java_com_baidu_paddle_inference_Config_fractionOfGpuMemoryForPool(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  float fraction_of_gpuMemory_for_pool = PD_ConfigFractionOfGpuMemoryForPool(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  return (jfloat)fraction_of_gpuMemory_for_pool;
}

// 7. TensorRT To Do

// 8. optim setting

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_switchIrOptim(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer, jboolean flag) {
  PD_ConfigSwitchIrOptim(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer),
                         jboolean_to_cpp_bool(env, flag));
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_inference_Config_irOptim(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  bool flag =
      PD_ConfigIrOptim(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  return cpp_bool_to_jboolean(env, flag);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_switchIrDebug(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer, jboolean flag) {
  PD_ConfigSwitchIrDebug(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer),
                         jboolean_to_cpp_bool(env, flag));
}

// 9. enable memory optimization

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_enableMemoryOptim(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer, jboolean flag) {
  PD_ConfigEnableMemoryOptim(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer),
      jboolean_to_cpp_bool(env, flag));
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_inference_Config_memoryOptimEnabled(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  bool flag = PD_ConfigMemoryOptimEnabled(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  return cpp_bool_to_jboolean(env, flag);
}

// 10. profile setting

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_enableProfile(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  PD_ConfigEnableProfile(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
}

JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_inference_Config_profileEnabled(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  bool flag = PD_ConfigProfileEnabled(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  return cpp_bool_to_jboolean(env, flag);
}

// 11. log setting

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_disableGlogInfo(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  PD_ConfigDisableGlogInfo(
      reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
}

// 12. view config configuration

JNIEXPORT jstring JNICALL Java_com_baidu_paddle_inference_Config_summary(
    JNIEnv* env, jobject obj, jlong cppPaddleConfigPointer) {
  const char* summary =
      PD_ConfigSummary(reinterpret_cast<PD_Config*>(cppPaddleConfigPointer));
  jstring jSummary = env->NewStringUTF(summary);
  free(const_cast<char*>(summary));
  return jSummary;
}
