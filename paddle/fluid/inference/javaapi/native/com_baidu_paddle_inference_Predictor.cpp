#include <jni.h>
#include "pd_inference_api.h"
#include "com_baidu_paddle_inference_Predictor.h"

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Predictor_predictorTryShrinkMemory
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    PD_PredictorTryShrinkMemory((PD_Predictor*) cppPaddleConfigPointer);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Predictor_predictorClearIntermediateTensor
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    PD_PredictorClearIntermediateTensor((PD_Predictor*) cppPaddleConfigPointer);
}

JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Predictor_createPredictor
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    return (jlong) PD_PredictorCreate((PD_Config*) cppPaddleConfigPointer);
}

JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Predictor_getInputNum
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    return (jlong) PD_PredictorGetInputNum((PD_Predictor*) cppPaddleConfigPointer);
}


JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Predictor_getOutputNum
  (JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    return (jlong) PD_PredictorGetOutputNum((PD_Predictor*) cppPaddleConfigPointer);
}

JNIEXPORT jstring JNICALL Java_com_baidu_paddle_inference_Predictor_getInputNameByIndex
  (JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer, jlong index)
{
    const char *c_str = PD_PredictorGetInputNames((PD_Predictor*) cppPaddleConfigPointer)->data[(int) index];
    return env->NewStringUTF(c_str);
//    return (jstring) (PD_PredictorGetInputNames((PD_Predictor*) cppPaddleConfigPointer)->data[(int) index]);
}

JNIEXPORT jstring JNICALL Java_com_baidu_paddle_inference_Predictor_getOutputNameByIndex
  (JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer, jlong index)
{
    const char *c_str = PD_PredictorGetOutputNames((PD_Predictor*) cppPaddleConfigPointer)->data[(int) index];
    return env->NewStringUTF(c_str);
}

JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Predictor_getInputHandleByName
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer, jstring name)
{
    const char * input_name = env->GetStringUTFChars(name, 0);
    PD_Predictor* pd_predictor = (PD_Predictor*) cppPaddleConfigPointer;
    jlong output_tensor = (jlong)PD_PredictorGetInputHandle(pd_predictor, input_name);
    return output_tensor;
}

JNIEXPORT jlong JNICALL Java_com_baidu_paddle_inference_Predictor_getOutputHandleByName
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer, jstring name)
{
    const char * output_name = env->GetStringUTFChars(name, 0);
    PD_Predictor* pd_predictor = (PD_Predictor*) cppPaddleConfigPointer;
    jlong output_tensor = (jlong)PD_PredictorGetOutputHandle(pd_predictor, output_name);
    return output_tensor;
}

JNIEXPORT jboolean JNICALL Java_com_baidu_paddle_inference_Predictor_runPD
(JNIEnv * env, jobject obj, jlong cppPaddleConfigPointer)
{
    return (jboolean) PD_PredictorRun((PD_Predictor*) cppPaddleConfigPointer);
}
