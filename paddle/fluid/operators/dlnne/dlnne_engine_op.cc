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

#include "paddle/fluid/operators/dlnne/dlnne_engine_op.h"

namespace paddle {
namespace inference {

void CopyTensorDeviceToCpu(void* dst_ptr, void* src_ptr, int total_bytes) {
  cudaDeviceSynchronize();
  cudaMemcpy(dst_ptr, src_ptr, total_bytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
}
void CopyTensorCpuToDevice(void* dst_ptr, void* src_ptr, int total_bytes) {
  cudaDeviceSynchronize();
  cudaMemcpy(dst_ptr, src_ptr, total_bytes, cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
}

std::string ConvertType(paddle::experimental::DataType type) {
  switch (type) {
    case paddle::experimental::DataType::FLOAT32: {
      return "float32";
    }
    case paddle::experimental::DataType::INT64: {
      return "int64";
    }
    case paddle::experimental::DataType::INT32: {
      return "int32";
    }
    case paddle::experimental::DataType::FLOAT16: {
      return "float16";
    }
    default: {
      PADDLE_THROW(
          platform::errors::Fatal("The DLNNE Calibration only support "
                                  "float/float16/int32_t/int64_t input."));
    }
  }
}

int GetDataByte(paddle::experimental::DataType type) {
  switch (type) {
    case paddle::experimental::DataType::FLOAT32: {
      return 4;
    }
    case paddle::experimental::DataType::INT64: {
      return 8;
    }
    case paddle::experimental::DataType::INT32: {
      return 4;
    }
    case paddle::experimental::DataType::FLOAT16: {
      return 2;
    }
    default: {
      PADDLE_THROW(
          platform::errors::Fatal("The DLNNE Calibration only support "
                                  "float/float16/int32_t/int64_t input."));
    }
  }
}

std::string GenerateRandomKey() {
  std::string str(
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
  std::random_device rd;
  std::mt19937 generator(rd());

  std::shuffle(str.begin(), str.end(), generator);
  return str.substr(0, 32);
}

void ConvertPaddle2Onnx(std::string onnx_file_name,
                        std::string subgraph_root_path) {
  if (!FileExists(onnx_file_name.c_str())) {
    std::stringstream convert_cmd;
    convert_cmd << "paddle2onnx --model_dir " << subgraph_root_path
                << " --save_file " << onnx_file_name << " --opset_version 11";
    LOG(INFO) << convert_cmd.str();
    int convert_flag = system(convert_cmd.str().c_str());
    PADDLE_ENFORCE_EQ(
        convert_flag,
        0,
        platform::errors::Unavailable("Convert paddle to onnx failed"));
  }
}

void QuantizeOnnx(std::string onnx_file_name,
                  std::string rlym_file_name,
                  std::string quantized_rlym_file_name,
                  std::string dataset_path,
                  std::string dataset_plugin_path) {
  if (!FileExists(rlym_file_name.c_str())) {
    std::stringstream convert_cmd;
    convert_cmd << "python -m dl convert " << onnx_file_name
                << " --output-model " << rlym_file_name;
    LOG(INFO) << convert_cmd.str();
    int convert_flag = system(convert_cmd.str().c_str());
    PADDLE_ENFORCE_EQ(
        convert_flag,
        0,
        platform::errors::Unavailable("Convert onnx to rlym failed"));
  }

  if (!FileExists(quantized_rlym_file_name.c_str())) {
    std::stringstream quantize_cmd;
    quantize_cmd << "python -m dl quantize "
                 << "--dataset " << dataset_path << " --plugin "
                 << dataset_plugin_path << " " << rlym_file_name;
    LOG(INFO) << quantize_cmd.str();
    int quantize_flag = system(quantize_cmd.str().c_str());
    PADDLE_ENFORCE_EQ(quantize_flag,
                      0,
                      platform::errors::Unavailable("quantize model failed"));
  }
}

}  // namespace inference

namespace operators {

class DlnneEngineOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Xs", "A list of inputs.").AsDuplicable();
    AddOutput("Ys", "A list of outputs").AsDuplicable();
    AddAttr<std::string>("subgraph", "the subgraph.");
    AddAttr<std::string>(
        "engine_key",
        "The engine_key here is used to distinguish different DLNNE Engines");
    AddAttr<int32_t>("max_batch_size", "engine max_batch_size");
    AddAttr<bool>("use_static_batch", "static batch fix for [?,H,W,C]");
    AddAttr<std::string>("weight_share_mode",
                         "dlnne weight_share_mode, can be '0', '1', '2', '3', "
                         "'01', '23', '0123' ");
    // when use_calib_mode is true and enable_int8 is true,
    // the calibration_runtime start,
    // when calibration_mode is true, the calibration_runtiime
    // go to the first stage of calibration, and when finish
    // fisrt stage, the calibration_mode is set false, the
    // calibration_runtime go to the second stage
    AddAttr<bool>("use_calib_mode", "dlnne use calib mode");
    AddAttr<bool>("enable_int8", "dlnne enable int8");
    AddAttr<bool>("calibration_mode", "dlnne calibration_mode");
    AddAttr<std::string>("calibration_data_path", "calibration data path");
    AddAttr<std::string>("subgraph_root_path", "subgraph root path");
    AddAttr<framework::BlockDesc*>("sub_block", "the dlnne block");
    AddComment("Dlnne engine operator.");
  }
};

class DlnneEngineInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OPERATOR(dlnne_engine, ops::DlnneEngineOp, ops::DlnneEngineOpMaker);
