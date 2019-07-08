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

public class Tensor {

    private long cppTensorPointer;
    private boolean readOnly;
    private PaddlePredictor predictor;

    /**
     * Accessed by package only to prevent public users to create it wrongly. A
     * Tensor can be created by {@link com.baidu.paddle.lite.PaddlePredictor} only
     */
    protected Tensor(long cppTensorPointer, boolean readOnly, PaddlePredictor predictor) {
        this.cppTensorPointer = cppTensorPointer;
        this.readOnly = readOnly;
        this.predictor = predictor;
    }

    protected void finalize() throws Throwable {
        if (cppTensorPointer != 0L) {
            deleteCppTensor(cppTensorPointer);
            cppTensorPointer = 0L;
        }
        super.finalize();
    }

    public boolean isReadOnly() {
        return readOnly;
    }

    public native boolean resize(long[] dims);

    public native long[] shape();

    public native boolean setData(float[] buf);

    public native boolean setData(byte[] buf);

    public native float[] getFloatData();

    public native byte[] getByteData();

    private native boolean deleteCppTensor(long native_pointer);
}