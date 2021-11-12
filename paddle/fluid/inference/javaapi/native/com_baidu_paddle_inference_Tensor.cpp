#include <jni.h>
#include "pd_inference_api.h"
#include "com_baidu_paddle_inference_Tensor.h"

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_TensorReshape
(JNIEnv *env, jobject, jlong tensorPointer, jint dim, jintArray array)
{
    int32_t *input_shape = env->GetIntArrayElements(array, nullptr);
    PD_TensorReshape((PD_Tensor*)tensorPointer, (int)dim, input_shape);
}


JNIEXPORT jintArray JNICALL Java_com_baidu_paddle_inference_Tensor_TensorGetShape
(JNIEnv *env, jobject, jlong tensorPointer)
{
    PD_Tensor* tensor = (PD_Tensor*)tensorPointer;
    PD_OneDimArrayInt32* output_shape = PD_TensorGetShape(tensor);
    jintArray result = env->NewIntArray(output_shape->size);
    env->SetIntArrayRegion(result, 0, output_shape->size, output_shape->data);
    return result;
}

JNIEXPORT jstring JNICALL Java_com_baidu_paddle_inference_Tensor_TensorGetName
(JNIEnv *env, jobject, jlong tensorPointer)
{
    const char *c_str = PD_TensorGetName((PD_Tensor*) tensorPointer);
    return env->NewStringUTF(c_str);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_TensorCopyFromCpuFloat
(JNIEnv *env, jobject, jlong tensorPointer, jfloatArray array)
{
    float *data = env->GetFloatArrayElements(array, nullptr);
    PD_TensorCopyFromCpuFloat((PD_Tensor*)tensorPointer, data);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_TensorCopyFromCpuInt
(JNIEnv *env, jobject, jlong tensorPointer, jintArray array)
{
    int32_t *data = env->GetIntArrayElements(array, nullptr);
    PD_TensorCopyFromCpuInt32((PD_Tensor*)tensorPointer, data);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_TensorCopyFromCpuLong
(JNIEnv *env, jobject, jlong tensorPointer, jlongArray array)
{
    int64_t *data = env->GetLongArrayElements(array, nullptr);
    PD_TensorCopyFromCpuInt64((PD_Tensor*)tensorPointer, data);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_TensorCopyFromCpuByte
(JNIEnv *env, jobject, jlong tensorPointer, jbyteArray array)
{
    int8_t *data = env->GetByteArrayElements(array, nullptr);
    PD_TensorCopyFromCpuInt8((PD_Tensor*)tensorPointer, data);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_TensorCopyFromCpuBoolean
(JNIEnv *env, jobject, jlong tensorPointer, jbooleanArray array)
{
    uint8_t *data = env->GetBooleanArrayElements(array, nullptr);
    PD_TensorCopyFromCpuUint8((PD_Tensor*)tensorPointer, data);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_TensorCopyToCpuFloat
(JNIEnv *env, jobject, jlong tensorPointer, jfloatArray array)
{
    float *data = env->GetFloatArrayElements(array, nullptr);
    PD_TensorCopyToCpuFloat((PD_Tensor*)tensorPointer, data);
    env->ReleaseFloatArrayElements(array, data, 0);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_TensorCopyToCpuInt
(JNIEnv *env, jobject, jlong tensorPointer, jintArray array)
{
    int32_t *data = env->GetIntArrayElements(array, nullptr);
    PD_TensorCopyToCpuInt32((PD_Tensor*)tensorPointer, data);
    env->ReleaseIntArrayElements(array, data, 0);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_TensorCopyToCpuLong
(JNIEnv *env, jobject, jlong tensorPointer, jlongArray array)
{
    int64_t *data = env->GetLongArrayElements(array, nullptr);
    PD_TensorCopyToCpuInt64((PD_Tensor*)tensorPointer, data);
    env->ReleaseLongArrayElements(array, data, 0);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_TensorCopyToCpuByte
(JNIEnv *env, jobject, jlong tensorPointer, jbyteArray array)
{
    int8_t *data = env->GetByteArrayElements(array, nullptr);
    PD_TensorCopyToCpuInt8((PD_Tensor*)tensorPointer, data);
    env->ReleaseByteArrayElements(array, data, 0);
}

JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_TensorCopyToCpuBoolean
(JNIEnv *env, jobject, jlong tensorPointer, jbooleanArray array)
{
    uint8_t *data = env->GetBooleanArrayElements(array, nullptr);
    PD_TensorCopyToCpuUint8((PD_Tensor*)tensorPointer, data);
    env->ReleaseBooleanArrayElements(array, data, 0);
}
