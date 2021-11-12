#include <iostream>
#include "jni_convert_util.h"
#include "com_baidu_paddle_inference_Config.h"

#include "pd_inference_api.h"

// 1. create Config

JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Config_createCppConfig
(JNIEnv * env, jobject obj)
{
    jlong cppPaddleConfigPointer = (jlong) PD_ConfigCreate();
    return cppPaddleConfigPointer;
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_inference_Config_isCppConfigValid
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    bool flag = PD_ConfigIsValid((PD_Config*)cppPaddleConfigPointer);
    return cpp_bool_to_jboolean(env, flag);
}

// 2. not combined model settings

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_setCppModel
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer, jstring modelFile, jstring paramsFile)
{
    const char * model_file = env->GetStringUTFChars(modelFile, 0);
    const char * params_file =  env->GetStringUTFChars(paramsFile, 0);
    PD_ConfigSetModel((PD_Config*) cppPaddleConfigPointer, model_file, params_file);
    free(const_cast<char*>(model_file));
    free(const_cast<char*>(params_file));
}

// 3. combined model settings

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_setCppModelDir
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer, jstring modelDir)
{
    const char * model_dir = env->GetStringUTFChars(modelDir, 0);
    PD_ConfigSetModelDir((PD_Config*)cppPaddleConfigPointer, model_dir);
    free(const_cast<char*>(model_dir));
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_setCppProgFile
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer, jstring progFile)
{
    const char * prog_file = env->GetStringUTFChars(progFile, 0);
    PD_ConfigSetProgFile((PD_Config*)cppPaddleConfigPointer, prog_file);
    free(const_cast<char*>(prog_file));
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_setCppParamsFile
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer, jstring paramsFile)
{
    const char * params_file =  env->GetStringUTFChars(paramsFile, 0);
    PD_ConfigSetParamsFile((PD_Config*)cppPaddleConfigPointer, params_file);
    free(const_cast<char*>(params_file));
}

JNIEXPORT jstring JNICALL Java_com_baidu_paddle_inference_Config_modelDir
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    const char * model_dir = PD_ConfigGetModelDir((PD_Config*)cppPaddleConfigPointer);
    jstring modelDir = env->NewStringUTF(model_dir);
    free(const_cast<char*>(model_dir));
    return modelDir;
}

JNIEXPORT jstring JNICALL Java_com_baidu_paddle_inference_Config_progFile
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    const char * prog_file = PD_ConfigGetProgFile((PD_Config*)cppPaddleConfigPointer);
    jstring progFile = env->NewStringUTF(prog_file);
    free(const_cast<char*>(prog_file));
    return progFile;
}

JNIEXPORT jstring JNICALL Java_com_baidu_paddle_inference_Config_paramsFile
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    const char * params_file = PD_ConfigGetProgFile((PD_Config*)cppPaddleConfigPointer);
    jstring paramsFile = env->NewStringUTF(params_file);
    free(const_cast<char*>(params_file));
    return paramsFile;
}

// 4. cpu settings

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_setCpuMathLibraryNumThreads
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer, jint mathThreadsNum)
{
    int math_threads_num = (int) mathThreadsNum;
    PD_ConfigSetCpuMathLibraryNumThreads((PD_Config*)cppPaddleConfigPointer, math_threads_num);
}

JNIEXPORT jint JNICALL Java_com_baidu_paddle_inference_Config_cpuMathLibraryNumThreads
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    jint mathThreadsNum = (jint) PD_ConfigGetCpuMathLibraryNumThreads((PD_Config*)cppPaddleConfigPointer);
}

// 5. MKLDNN settings

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_enableMKLDNN
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    PD_ConfigEnableMKLDNN((PD_Config*)cppPaddleConfigPointer);
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_inference_Config_mkldnnEnabled
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    bool flag = PD_ConfigMkldnnEnabled((PD_Config*)cppPaddleConfigPointer);
    return cpp_bool_to_jboolean(env, flag);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_enableMkldnnBfloat16
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    PD_ConfigEnableMkldnnBfloat16((PD_Config*)cppPaddleConfigPointer);
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_inference_Config_mkldnnBfloat16Enabled
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    bool flag = PD_ConfigMkldnnBfloat16Enabled((PD_Config*)cppPaddleConfigPointer);
    return cpp_bool_to_jboolean(env, flag);
}

// 6. gpu setting

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_enableUseGpu
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer, jlong memorySize, jint deviceId)
{
    PD_ConfigEnableUseGpu((PD_Config*)cppPaddleConfigPointer, (uint64_t) memorySize, (int32_t) deviceId);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_disableGpu
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    PD_ConfigDisableGpu((PD_Config*)cppPaddleConfigPointer);
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_inference_Config_useGpu
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    bool flag = PD_ConfigUseGpu((PD_Config*)cppPaddleConfigPointer);
    return cpp_bool_to_jboolean(env, flag);
}

JNIEXPORT jint JNICALL Java_com_baidu_paddle_inference_Config_gpuDeviceId
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    int device_id = PD_ConfigGpuDeviceId((PD_Config*)cppPaddleConfigPointer);
    return (jint) device_id;
}

JNIEXPORT jint JNICALL Java_com_baidu_paddle_inference_Config_memoryPoolInitSizeMb
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    int memory_pool_init_size_mb = PD_ConfigMemoryPoolInitSizeMb((PD_Config*)cppPaddleConfigPointer);
    return (jint) memory_pool_init_size_mb;
}

JNIEXPORT jfloat JNICALL Java_com_baidu_paddle_inference_Config_fractionOfGpuMemoryForPool
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    float fraction_of_gpuMemory_for_pool = PD_ConfigFractionOfGpuMemoryForPool((PD_Config*)cppPaddleConfigPointer);
    return (jfloat) fraction_of_gpuMemory_for_pool;
}

// 7. TensorRT To Do

// 8. optim setting

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_switchIrOptim
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer, jboolean flag)
{
    PD_ConfigSwitchIrOptim((PD_Config*)cppPaddleConfigPointer, jboolean_to_cpp_bool(env, flag));
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_inference_Config_irOptim
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    bool flag = PD_ConfigIrOptim((PD_Config*)cppPaddleConfigPointer);
    return cpp_bool_to_jboolean(env, flag);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_switchIrDebug
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer, jboolean flag)
{
    PD_ConfigSwitchIrDebug((PD_Config*)cppPaddleConfigPointer, jboolean_to_cpp_bool(env, flag));
}

// 9. enable memory optimization

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_enableMemoryOptim
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer, jboolean flag)
{
    PD_ConfigEnableMemoryOptim((PD_Config*)cppPaddleConfigPointer, jboolean_to_cpp_bool(env, flag));
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_inference_Config_memoryOptimEnabled
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    bool flag = PD_ConfigMemoryOptimEnabled((PD_Config*)cppPaddleConfigPointer);
    return cpp_bool_to_jboolean(env, flag);
}

// 10. profile setting

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_enableProfile
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    PD_ConfigEnableProfile((PD_Config*)cppPaddleConfigPointer);
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_inference_Config_profileEnabled
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    bool flag = PD_ConfigProfileEnabled((PD_Config*)cppPaddleConfigPointer);
    return cpp_bool_to_jboolean(env, flag);
}

// 11. log setting

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Config_disableGlogInfo
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    PD_ConfigDisableGlogInfo((PD_Config*)cppPaddleConfigPointer);
}

// 12. view config configuration

JNIEXPORT jstring JNICALL Java_com_baidu_paddle_inference_Config_summary
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    const char * summary = PD_ConfigSummary((PD_Config*)cppPaddleConfigPointer);
    jstring jSummary = env->NewStringUTF(summary);
    free(const_cast<char*>(summary));
    return jSummary;
}







