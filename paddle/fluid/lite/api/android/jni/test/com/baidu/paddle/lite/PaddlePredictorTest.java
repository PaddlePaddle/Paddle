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

import org.junit.jupiter.api.Test;

import static org.junit.Assert.assertEquals;

class PaddlePredictorTest {

    @Test
    public void run_defaultModel() {
        MobileConfig config = new MobileConfig();
        config.setModelDir("");
        PaddlePredictor predictor = PaddlePredictor.createPaddlePredictor(config);

        float[] inputBuffer = new float[10000];
        for (int i = 0; i < 10000; ++i) {
            inputBuffer[i] = i;
        }
        long[] dims = { 100, 100 };

        Tensor input = predictor.getInput(0);
        input.resize(dims);
        input.setData(inputBuffer);

        predictor.run();
        
        Tensor output = predictor.getOutput(0);
        float[] outputBuffer = output.getFloatData();

        assertEquals(outputBuffer.length, 50000);
        assertEquals(outputBuffer[0], 50.2132f, 1e-3f);
        assertEquals(outputBuffer[1], -28.8729f, 1e-3f);
    }

}
