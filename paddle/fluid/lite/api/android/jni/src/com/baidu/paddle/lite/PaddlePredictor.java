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

    /** 
     * Java doesn't have pointer. To maintain the life cycle of under going
     * C++ PaddlePredictor object, we use a long value to maintain it.
     */
    private long cppPaddlePredictorPointer;

    private PaddlePredictor(ConfigBase config) {
        init(config);
    }

    public static PaddlePredictor createPaddlePredictor(ConfigBase config) {
        PaddlePredictor predictor = new PaddlePredictor(config);
        return predictor.cppPaddlePredictorPointer == 0L ? null : predictor;
    }

    public Tensor getInput(int offset) {
        long cppTensorPointer = getInputCppTensorPointer(offset);
        return cppTensorPointer == 0 ? null : new Tensor(cppTensorPointer, /* readOnly = */ false, this);
    }

    public Tensor getOutput(int offset) {
        long cppTensorPointer = getOutputCppTensorPointer(offset);
        return cppTensorPointer == 0 ? null : new Tensor(cppTensorPointer, /* readOnly = */ true, this);
    }

    public Tensor getTensor(String name) {
        long cppTensorPointer = getCppTensorPointerByName(name);
        return cppTensorPointer == 0 ? null : new Tensor(cppTensorPointer, /* readOnly = */ true, this); 
    }

    public native boolean run();

    public native boolean saveOptimizedModel(String modelDir);

    @Override
    protected void finalize() throws Throwable {
        clear();
        super.finalize();
    }
    
    protected boolean init(ConfigBase config) {
        if (config instanceof CxxConfig) {
            cppPaddlePredictorPointer = newCppPaddlePredictor((CxxConfig)config);
        } else if (config instanceof MobileConfig) {
            cppPaddlePredictorPointer = newCppPaddlePredictor((MobileConfig)config);
        } else {
            throw new IllegalArgumentException("Not supported PaddleLite Config type");
        }
        return cppPaddlePredictorPointer != 0L;
    }

    protected boolean clear() {
        boolean result = false;
        if (cppPaddlePredictorPointer != 0L) {
            result = deleteCppPaddlePredictor(cppPaddlePredictorPointer);
            cppPaddlePredictorPointer = 0L;
        }
        return result;
    }

    private native long getInputCppTensorPointer(int offset);

    private native long getOutputCppTensorPointer(int offset);
    
    private native long getCppTensorPointerByName(String name);

    private native long newCppPaddlePredictor(CxxConfig config);

    private native long newCppPaddlePredictor(MobileConfig config);

    private native boolean deleteCppPaddlePredictor(long nativePointer);

    static {
        PaddleLiteInitializer.init();
    }
}
