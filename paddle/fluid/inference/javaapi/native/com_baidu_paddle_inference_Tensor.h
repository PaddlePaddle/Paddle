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

/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_baidu_paddle_inference_Tensor */

#ifndef _Included_com_baidu_paddle_inference_Tensor  // NOLINT
#define _Included_com_baidu_paddle_inference_Tensor
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_baidu_paddle_inference_Tensor
 * Method:    TensorReshape
 * Signature: (JI[I)V
 */
JNIEXPORT void JNICALL Java_com_baidu_paddle_inference_Tensor_TensorReshape(
    JNIEnv *, jobject, jlong, jint, jintArray);

/*
 * Class:     com_baidu_paddle_inference_Tensor
 * Method:    TensorGetShape
 * Signature: (J)[I
 */
JNIEXPORT jintArray JNICALL
Java_com_baidu_paddle_inference_Tensor_TensorGetShape(JNIEnv *, jobject, jlong);

/*
 * Class:     com_baidu_paddle_inference_Tensor
 * Method:    TensorGetName
 * Signature: (J)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL
Java_com_baidu_paddle_inference_Tensor_TensorGetName(JNIEnv *, jobject, jlong);

/*
 * Class:     com_baidu_paddle_inference_Tensor
 * Method:    TensorCopyFromCpuFloat
 * Signature: (J[F)V
 */
JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_TensorCopyFromCpuFloat(JNIEnv *, jobject,
                                                              jlong,
                                                              jfloatArray);

/*
 * Class:     com_baidu_paddle_inference_Tensor
 * Method:    TensorCopyFromCpuInt
 * Signature: (J[I)V
 */
JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_TensorCopyFromCpuInt(JNIEnv *, jobject,
                                                            jlong, jintArray);

/*
 * Class:     com_baidu_paddle_inference_Tensor
 * Method:    TensorCopyFromCpuLong
 * Signature: (J[J)V
 */
JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_TensorCopyFromCpuLong(JNIEnv *, jobject,
                                                             jlong, jlongArray);

/*
 * Class:     com_baidu_paddle_inference_Tensor
 * Method:    TensorCopyFromCpuByte
 * Signature: (J[B)V
 */
JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_TensorCopyFromCpuByte(JNIEnv *, jobject,
                                                             jlong, jbyteArray);

/*
 * Class:     com_baidu_paddle_inference_Tensor
 * Method:    TensorCopyFromCpuBoolean
 * Signature: (J[Z)V
 */
JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_TensorCopyFromCpuBoolean(JNIEnv *,
                                                                jobject, jlong,
                                                                jbooleanArray);

/*
 * Class:     com_baidu_paddle_inference_Tensor
 * Method:    TensorCopyToCpuFloat
 * Signature: (J[F)V
 */
JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_TensorCopyToCpuFloat(JNIEnv *, jobject,
                                                            jlong, jfloatArray);

/*
 * Class:     com_baidu_paddle_inference_Tensor
 * Method:    TensorCopyToCpuInt
 * Signature: (J[I)V
 */
JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_TensorCopyToCpuInt(JNIEnv *, jobject,
                                                          jlong, jintArray);

/*
 * Class:     com_baidu_paddle_inference_Tensor
 * Method:    TensorCopyToCpuLong
 * Signature: (J[J)V
 */
JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_TensorCopyToCpuLong(JNIEnv *, jobject,
                                                           jlong, jlongArray);

/*
 * Class:     com_baidu_paddle_inference_Tensor
 * Method:    TensorCopyToCpuByte
 * Signature: (J[B)V
 */
JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_TensorCopyToCpuByte(JNIEnv *, jobject,
                                                           jlong, jbyteArray);

/*
 * Class:     com_baidu_paddle_inference_Tensor
 * Method:    TensorCopyToCpuBoolean
 * Signature: (J[Z)V
 */
JNIEXPORT void JNICALL
Java_com_baidu_paddle_inference_Tensor_TensorCopyToCpuBoolean(JNIEnv *, jobject,
                                                              jlong,
                                                              jbooleanArray);

#ifdef __cplusplus
}
#endif
#endif  // NOLINT
