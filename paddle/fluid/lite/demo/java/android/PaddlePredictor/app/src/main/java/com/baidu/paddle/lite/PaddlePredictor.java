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

package com.baidu.paddle.lite;

/** Java Native Interface (JNI) class for Paddle Lite APIs */
public class PaddlePredictor {

    /** name of C++ JNI lib */
    private final static String JNI_LIB_NAME = "paddle_lite_jni";

    /* load the C++ JNI lib */
    static {
        System.loadLibrary(JNI_LIB_NAME);
    }

    /**
     * Loads mobile cxx model, which is the model before optimizing passes. The cxx
     * model allow users to manage hardware place resources. Caller uses a place at
     * Java to control Target, DataLayout, Precision, and Device ID. More details
     * about the four fields see our Paddle-Mobile document.
     *
     *
     * @param modelPath      modelPath model file path
     * @param preferredPlace preferred place to run Cxx Model
     * @param validPlaces    n * 4 int array, valid places to run Cxx Model
     * @return true if load successfully
     */
    public static native boolean loadCxxModel(String modelPath, Place preferredPlace, Place[] validPlaces);

    /**
     * Loads mobile lite model, which is the model after optimizing passes.
     *
     * @param modelPath model file path
     * @return true if load successfully
     */
    public static native boolean loadMobileModel(String modelPath);

    /**
     * Saves optimized model, which is the model can be used by
     * {@link loadMobileModel}
     *
     * @param modelPath model file path
     * @return true if save successfully
     */
    public static native boolean saveOptimizedModel(String modelPath);

    /**
     * Clears the current loaded model.
     *
     * @return true if a loaded model has been cleared.
     */
    public static native boolean clear();

    /**
     * Set input data on offset-th column of feed data
     *
     * @param offset the offset-th column of feed data will be set
     * @param buf    the input data
     * @param dims   dimension format of the input image
     * @return true if set successfully
     */
    public static native boolean setInput(int offset, int[] dims, float[] buf);

    /**
     * Set input data on offset-th column of feed data
     *
     * @param offset the offset-th column of feed data will be set
     * @param buf    the input data
     * @param dims   dimension format of the input image
     * @return true if set successfully
     */
    public static native boolean setInput(int offset, int[] dims, byte[] buf);

    /**
     * Run the predict model
     *
     * @return true if run successfully
     */
    public static native boolean run();

    /**
     * Get offset-th column of output data as float
     *
     * @param offset the offset-th column of output data will be returned
     * @return model predict output
     */
    public static native float[] getFloatOutput(int offset);

    /**
     * Get offset-th column of output data as byte (int8 in C++ side)
     *
     * @param offset the offset-th column of output data will be returned
     * @return model predict output
     */
    public static native byte[] getByteOutput(int offset);

    /**
     * Fetches a Tensor's value as Float data
     *
     * @param name Tensor's name
     * @return values of the Tensor
     */
    public static native float[] fetchFloat(String name);

    /**
     * Fetches a Tensor's value as byte data (int8 at C++ side)
     *
     * @param name Tensor's name
     * @return values of the Tensor
     */
    public static native byte[] fetchByte(String name);

    /**
     * Main function for test
     */
    public static void main(String[] args) {
        System.out.println("Load native library successfully");
    }
}
