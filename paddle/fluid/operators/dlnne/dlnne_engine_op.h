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

#pragma once
#include <cuda.h>          // NOTLINT
#include <cuda_runtime.h>  // NOTLINT
#include <dlnne.h>         // NOTLINT

#include <assert.h>
#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/inference/analysis/helper.h"

namespace dl {
namespace nne {
class Builder;
class Engine;
class Network;
class Parser;
class ExecutionContext;
}  // namespace nne
}  // namespace dl

namespace paddle {
namespace inference {
class NneDeleter {
 public:
  NneDeleter() {}

  template <typename T>
  inline void operator()(T *ptr) {
    if (ptr != nullptr) {
      ptr->Destroy();
    }
  }
};

void CopyTensorDeviceToCpu(void *dst_ptr, void *src_ptr, int total_bytes);

void CopyTensorCpuToDevice(void *dst_ptr, void *src_ptr, int total_bytes);

template <typename T>
struct Singleton;
}  // namespace inference
}  // namespace paddle

namespace paddle {

namespace operators {

class DlnneEngineOp : public framework::OperatorBase {
 private:
  std::vector<std::string> input_names_;
  std::unordered_set<std::string> param_names_;
  std::string engine_key_;
  int num_inputs;
  int num_outputs;
  std::vector<std::string> output_names;
  std::vector<std::string> input_names;

  dl::nne::Builder *builder;
  dl::nne::Parser *parser;
  dl::nne::Network *network;
  dl::nne::ExecutionContext *context;
  dl::nne::Engine *engine;

  unsigned int engine_input_size;
  std::vector<int> InputIndexToBindIndex_;

 public:
  DlnneEngineOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {
    input_names_ = Inputs("Xs");
    engine_key_ = Attr<std::string>("engine_key");
    auto params = Attr<std::vector<std::string>>("parameters");
    for (const auto &param : params) {
      param_names_.insert(param);
    }

    num_inputs = 0;
    for (const auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      num_inputs += 1;
      input_names.push_back(x);
    }

    num_outputs = Outputs("Ys").size();
    for (const auto &y : Outputs("Ys")) {
      VLOG(4) << "y: " << y << std::endl;
      output_names.push_back(y);
    }

    // onnx path
    std::stringstream filename;
    std::string current_path = ".";
    char *buffer;
    if ((buffer = getcwd(NULL, 0)) != NULL) {
      current_path = buffer;
    } else {
      current_path = ".";
    }
    filename << current_path << "/dump/" << engine_key_ << "/" << engine_key_
             << ".onnx";

    builder = dl::nne::CreateInferBuilder();
    PADDLE_ENFORCE_NE(builder, nullptr, platform::errors::Unavailable(
                                            "nne create builder failed"));
    parser = dl::nne::CreateParser();
    PADDLE_ENFORCE_NE(parser, nullptr, platform::errors::Unavailable(
                                           "nne create parser failed"));

    network = builder->CreateNetwork();

    LOG(INFO) << "set output for dlnne";
    for (std::string &output_op_name : output_names)
      parser->RegisterOutput(output_op_name.c_str());

    LOG(INFO) << "parser onnx for dlnne";
    parser->Parse(filename.str().c_str(), *network);

    LOG(INFO) << "build network";
    engine = builder->BuildEngine(*network);

    // total size = input_size+output_size
    engine_input_size = num_inputs + num_outputs;
    for (std::string &input_name : input_names) {
      int BindIndex = engine->GetBindingIndex(input_name.c_str());
      InputIndexToBindIndex_.push_back(BindIndex);
    }

    for (std::string &output_name : output_names) {
      int BindIndex = engine->GetBindingIndex(output_name.c_str());
      InputIndexToBindIndex_.push_back(BindIndex);
    }

    // context
    context = engine->CreateExecutionContext();
  }

  ~DlnneEngineOp() {
    network->Destroy();
    context->Destroy();
    engine->Destroy();
    parser->Destroy();
    builder->Destroy();
  }

 protected:
  void RunDlnneOnCreateEngine(const framework::Scope &scope,
                              const platform::Place &dev_place) const {
    PADDLE_ENFORCE_EQ(
        input_names_.empty(), false,
        platform::errors::PreconditionNotMet(
            "Dlnne engine needs at least one input, but no input is found. "
            "Please check if you set the input correctly."));

    std::vector<void *> input_buffers(num_inputs);
    std::vector<void *> cpu_input_buffers(num_inputs);
    std::vector<std::vector<int64_t>> input_shapes(num_inputs);
    std::vector<int32_t> input_data_types(num_inputs);
    std::vector<int64_t> input_bytes(num_inputs);

    int index = 0;
    for (const auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      // convert input and copy to Dlnne engine's buffer
      auto &t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);

      const int bind_index = index;
      index++;
      int64_t data_bytes;
      int32_t dtype;
      auto type = framework::TransToProtoVarType(t.dtype());
      data_bytes = 1;
      void *buffer = nullptr;
      if (type == framework::proto::VarType::FP32) {
        buffer = static_cast<void *>(t.data<float>());
        data_bytes = 4;
        dtype = 0;
      } else if (type == framework::proto::VarType::INT64) {
        buffer = static_cast<void *>(t.data<int64_t>());
        data_bytes = 8;
        dtype = 1;
      } else if (type == framework::proto::VarType::INT32) {
        buffer = static_cast<void *>(t.data<int32_t>());
        data_bytes = 4;
        dtype = 2;
      } else {
        PADDLE_THROW(platform::errors::Fatal(
            "The DLNNE Engine OP only support float/int32_t/int64_t input."));
      }
      input_buffers[bind_index] = buffer;

      auto t_shape = framework::vectorize<int64_t>(t.dims());
      std::vector<int64_t> runtime_input_shape(t_shape.begin(), t_shape.end());
      for (auto &size : t_shape) {
        data_bytes = data_bytes * size;
      }

      VLOG(4) << "buffers_size:" << data_bytes;
      cpu_input_buffers[bind_index] =
          input_buffers[bind_index];  // malloc(data_bytes);
      input_shapes[bind_index] = runtime_input_shape;
      input_data_types[bind_index] = dtype;
      input_bytes[bind_index] = data_bytes;
    }

    // output shape
    std::vector<std::vector<int64_t>> out_shapes;
    std::vector<int32_t> output_bytes;
    for (int i = 0; i < num_outputs; i++) {
      int index = engine->GetBindingIndex(output_names[i].c_str());
      dl::nne::Dims out_dim = engine->GetBindingDimensions(index);
      std::vector<int64_t> shape(out_dim.nbDims);
      for (int dim = 0; dim < out_dim.nbDims; dim++) {
        shape[dim] = (out_dim.d[dim]);
      }

      out_shapes.push_back(shape);
      int64_t data_bytes;

      // float32
      data_bytes = 4;
      for (auto &size : shape) {
        data_bytes = data_bytes * size;
      }
      VLOG(4) << "data_bytes: " << data_bytes;
      output_bytes.push_back(data_bytes);
    }

    int bind_index = 0;
    std::vector<void *> cpu_output_buffers(num_outputs);
    std::vector<void *> output_buffers(num_outputs);
    std::vector<int32_t> output_dtypes(num_outputs);

    for (const auto &y : Outputs("Ys")) {
      auto *fluid_v = scope.FindVar(y);
      PADDLE_ENFORCE_NOT_NULL(
          fluid_v,
          platform::errors::NotFound(
              "Output variable %s is not found in DLNNE subgraph.", y));

      auto *fluid_t = fluid_v->GetMutable<framework::LoDTensor>();

      VLOG(4) << "out_shapes[bind_index] dim:" << out_shapes[bind_index].size();
      fluid_t->Resize(framework::make_ddim(out_shapes[bind_index]));

      int32_t dtype;
      output_buffers[bind_index] = fluid_t->mutable_data<float>(dev_place);
      dtype = 0;
      cpu_output_buffers[bind_index] =
          output_buffers[bind_index];  // malloc(data_bytes);
      output_dtypes[bind_index] = dtype;
      bind_index++;
    }

    std::vector<void *> engine_input_ptr(engine_input_size);

    // set input_ptr
    for (unsigned int i = 0; i < engine_input_size; i++) {
      if (InputIndexToBindIndex_[i] < 0) continue;

      if (engine->BindingIsInput(InputIndexToBindIndex_[i])) {
        // copy cpu buffer to gpu buffer
        int64_t total_bytes;
        total_bytes = input_bytes[i];
        VLOG(4) << "input_bytes: " << total_bytes;

        void *gpu_ptr;
        cudaMalloc(&gpu_ptr, total_bytes);
        engine_input_ptr[InputIndexToBindIndex_[i]] = gpu_ptr;

        paddle::inference::CopyTensorCpuToDevice(
            gpu_ptr, reinterpret_cast<void *>(cpu_input_buffers[i]),
            total_bytes);

      } else {
        int64_t total_size;
        total_size = output_bytes[i - input_names.size()];
        VLOG(4) << "output_bytes: " << total_size;
        void *gpu_ptr;
        cudaMalloc(&gpu_ptr, total_size);
        engine_input_ptr[InputIndexToBindIndex_[i]] = gpu_ptr;
      }
    }

    clock_t startTime, endTime;
    startTime = clock();
    context->Execute(1, engine_input_ptr.data());
    endTime = clock();
    double during_ms =
        static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC * 1000;
    LOG(INFO) << "dlNNE execute time: " << during_ms << " ms";

    bind_index = 0;
    for (unsigned int i = 0; i < engine_input_size; i++) {
      if (InputIndexToBindIndex_[i] < 0) continue;

      if (i >= input_names.size()) {
        void *cpu_ptr = cpu_output_buffers[i - input_names.size()];
        int64_t size;
        size = output_bytes[i - input_names.size()];
        paddle::inference::CopyTensorDeviceToCpu(
            cpu_ptr, engine_input_ptr[InputIndexToBindIndex_[i]], size);
        // dtype: float32
        int32_t dtypes;
        dtypes = 0;

        cpu_output_buffers[bind_index] = cpu_ptr;
        output_dtypes[bind_index] = dtypes;
        bind_index++;
      }
      cudaFree(engine_input_ptr[InputIndexToBindIndex_[i]]);
    }
  }

  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    RunDlnneOnCreateEngine(scope, dev_place);
  }
};

}  // namespace operators
}  // namespace paddle
