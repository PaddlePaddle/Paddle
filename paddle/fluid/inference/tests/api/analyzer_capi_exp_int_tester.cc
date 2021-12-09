/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <vector>
#include "paddle/fluid/inference/capi_exp/pd_inference_api.h"
#include "paddle/fluid/inference/tests/api/tester_helper.h"

namespace paddle {
namespace inference {
namespace analysis {

void predictor_run() {
  std::string model_dir = FLAGS_infer_model;
  PD_Config* config = PD_ConfigCreate();
  PD_ConfigDisableGpu(config);
  PD_ConfigSetCpuMathLibraryNumThreads(config, 10);
  PD_ConfigSwitchIrDebug(config, TRUE);
  PD_ConfigSetModelDir(config, model_dir.c_str());
  PD_Predictor* predictor = PD_PredictorCreate(config);
  PD_OneDimArrayCstr* input_names = PD_PredictorGetInputNames(predictor);
  LOG(INFO) << "The inputs' size is: " << input_names->size;
  EXPECT_EQ(input_names->size, 2u);

  int32_t shape_0[4] = {1, 3, 224, 224};
  float data_0[1 * 3 * 224 * 224] = {0};
  PD_Tensor* input_0 = PD_PredictorGetInputHandle(predictor, "image");
  PD_TensorReshape(input_0, 4, shape_0);
  PD_TensorCopyFromCpuFloat(input_0, data_0);
  int32_t shape_1[2] = {1, 1};
  int64_t data_1[1] = {0};
  PD_Tensor* input_1 = PD_PredictorGetInputHandle(predictor, "label");
  PD_TensorReshape(input_1, 2, shape_1);
  PD_TensorCopyFromCpuInt64(input_1, data_1);

  LOG(INFO) << "Run Inference in CAPI encapsulation. ";
  EXPECT_TRUE(PD_PredictorRun(predictor));

  PD_OneDimArrayCstr* output_names = PD_PredictorGetOutputNames(predictor);
  LOG(INFO) << "output size is: " << output_names->size;
  for (size_t index = 0; index < output_names->size; ++index) {
    LOG(INFO) << "output[" << index
              << "]'s name is: " << output_names->data[index];
    PD_Tensor* output =
        PD_PredictorGetOutputHandle(predictor, output_names->data[index]);
    PD_OneDimArrayInt32* shape = PD_TensorGetShape(output);
    LOG(INFO) << "output[" << index << "]'s shape_size is: " << shape->size;
    int32_t out_size = 1;
    for (size_t i = 0; i < shape->size; ++i) {
      LOG(INFO) << "output[" << index << "]'s shape is: " << shape->data[i];
      out_size = out_size * shape->data[i];
    }
    float* out_data = new float[out_size];
    PD_TensorCopyToCpuFloat(output, out_data);
    LOG(INFO) << "output[" << index << "]'s DATA is: " << out_data[0];
    delete[] out_data;
    PD_OneDimArrayInt32Destroy(shape);
    PD_TensorDestroy(output);
  }
  PD_PredictorClearIntermediateTensor(predictor);
  PD_PredictorTryShrinkMemory(predictor);
  PD_OneDimArrayCstrDestroy(output_names);
  PD_TensorDestroy(input_1);
  PD_TensorDestroy(input_0);
  PD_OneDimArrayCstrDestroy(input_names);
  PD_PredictorDestroy(predictor);
}

#ifdef PADDLE_WITH_MKLDNN
TEST(PD_PredictorRun, predictor_run) { predictor_run(); }
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
