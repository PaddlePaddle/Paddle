/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once

#ifdef PADDLE_WITH_CUDA

#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/trt_int8_calibrator.h"

namespace paddle {

DECLARE_int32(tensorrt_engine_batch_size);

namespace operators {

using FluidDT = framework::proto::VarType_Type;
using TRT_DT = nvinfer1::DataType;

namespace {

TRT_DT FluidDataType2TRT(FluidDT type) {
  switch (type) {
    case FluidDT::VarType_Type_FP32:
      return TRT_DT::kFLOAT;
    case FluidDT::VarType_Type_INT32:
      return TRT_DT::kINT32;
    default:
      return TRT_DT::kINT32;
  }
  PADDLE_THROW("unkown type");
  return TRT_DT::kINT32;
}

nvinfer1::Dims Vec2TRT_Dims(const std::vector<int64_t> &shape) {
  PADDLE_ENFORCE_GT(shape.size(), 1UL,
                    "TensorRT' tensor input requires at least 2 dimensions");
  PADDLE_ENFORCE_LE(shape.size(), 4UL,
                    "TensorRT' tensor input requires at most 4 dimensions");
  PADDLE_ENFORCE(shape.size() == 4UL || shape.size() == 2UL);
  if (shape.size() == 4UL)
    return nvinfer1::DimsCHW(shape[1], shape[2], shape[3]);
  return nvinfer1::DimsCHW(shape[1], 1, 1);
}

}  // namespace

using inference::Singleton;
using inference::tensorrt::TRT_EngineManager;
using inference::tensorrt::TRTInt8Calibrator;
using inference::tensorrt::TRT_CalibratorRes;
using inference::tensorrt::TRT_CalibratorResManager;

class TensorRTEngineOp : public framework::OperatorBase {
 private:
  std::unique_ptr<TRTInt8Calibrator> calibrator_;
  std::string precision_mode_;
  std::string calibration_data_;
  std::string engine_name_;
  std::vector<std::string> input_names_;
  std::unordered_set<std::string> param_names_;
  int max_batch_size_;
  int workspace_size_;
  bool calibration_mode_;

 public:
  TensorRTEngineOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {
    precision_mode_ = Attr<std::string>("precision_mode");
    calibration_data_ = Attr<std::string>("calibration_data");
    engine_name_ = Attr<std::string>("engine_uniq_key");
    input_names_ = Inputs("Xs");
    max_batch_size_ = Attr<int>("max_batch_size");
    workspace_size_ = Attr<int>("workspace_size");

    auto params = Attr<std::vector<std::string>>("parameters");
    for (const auto &param : params) {
      param_names_.insert(param);
    }

    calibration_mode_ =
        (precision_mode_ == "INT8" && calibration_data_.size() == 0);

    if (precision_mode_ == "INT8" && calibration_data_.size()) {
      calibrator_.reset(new TRTInt8Calibrator(calibration_data_));
    }
  }

 protected:
  void RunNative(const framework::Scope &scope,
                 const platform::Place &dev_place) const {
    framework::NaiveExecutor executor(dev_place);
    auto *block = Attr<framework::BlockDesc *>("sub_block");
    auto *program = block->Program();
    auto *scope_ptr = const_cast<framework::Scope *>(&scope);
    executor.Prepare(scope_ptr, *program, block->ID(), false);
    executor.Run();
  }

  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    if (calibration_mode_ == true) {
      RunCalibration(scope, dev_place);
      return;
    }
    RunTrt(scope, dev_place);
  }

  void RunCalibration(const framework::Scope &scope,
                      const platform::Place &dev_place) const {
    // Create calibrator here.
    if (!Singleton<TRT_CalibratorResManager>::Global().Has(engine_name_)) {
      TRT_CalibratorRes *calib_res =
          Singleton<TRT_CalibratorResManager>::Global().Create(engine_name_);
      std::unordered_map<std::string, size_t> calib_buffers;
      for (auto &x : input_names_) {
        if (param_names_.count(x)) continue;
        // convert input and copy to TRT engine's buffer
        auto &t =
            inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);
        calib_buffers[x] = t.memory_size();
      }

      calib_res->calib_.reset(
          new TRTInt8Calibrator(calib_buffers, FLAGS_tensorrt_engine_batch_size,
                                engine_name_, dev_place));

      // Should add an comments here.
      calib_res->thr_.reset(new std::thread([&]() {
        VLOG(3) << "start the calib trt engine thread";
        Prepare(scope, dev_place, precision_mode_, calib_res->calib_.get());
      }));
    }

    TRTInt8Calibrator *temp_calibrator =
        Singleton<TRT_CalibratorResManager>::Global()
            .Get(engine_name_)
            ->calib_.get();
    std::unordered_map<std::string, void *> calib_data;
    for (auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      auto &t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);
      calib_data.emplace(x, t.data<void>());
    }

    temp_calibrator->setBatch(calib_data);
    RunNative(scope, dev_place);
  }

  void RunTrt(const framework::Scope &scope,
              const platform::Place &dev_place) const {
    if (!Singleton<TRT_EngineManager>::Global().HasEngine(engine_name_)) {
      Prepare(scope, dev_place, precision_mode_, calibrator_.get());
    }
    auto *engine = Singleton<TRT_EngineManager>::Global().Get(engine_name_);
    PADDLE_ENFORCE(!input_names_.empty(), "should pass more than one inputs");
    PADDLE_ENFORCE_LE(FLAGS_tensorrt_engine_batch_size, max_batch_size_);

    std::vector<std::string> output_maps =
        Attr<std::vector<std::string>>("output_name_mapping");

    // Convert input tensor from fluid to engine.
    for (const auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      // convert input and copy to TRT engine's buffer
      auto &t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);
      if (platform::is_cpu_place(t.place())) {
        engine->SetInputFromCPU(x, static_cast<const void *>(t.data<void>()),
                                t.memory_size());
      } else {
        engine->SetInputFromGPU(x, static_cast<const void *>(t.data<void>()),
                                t.memory_size());
      }
    }
    // Execute the engine.
    PADDLE_ENFORCE_GT(FLAGS_tensorrt_engine_batch_size, 0);
    engine->Execute(FLAGS_tensorrt_engine_batch_size);

    // Convert output tensor from engine to fluid
    int output_index = 0;
    VLOG(4) << "TensorRT Engine Op Outputs:";
    for (const auto &y : Outputs("Ys")) {
      VLOG(4) << y;
      // convert output and copy to fluid.
      nvinfer1::ITensor *trt_t = engine->GetITensor(output_maps[output_index]);
      auto dims = trt_t->getDimensions();
      // Use the output ITensor's dims to reshape the Fluid Tensor.
      // The ITensor doesn't contain the batch size dim.
      std::vector<int> ddim;
      ddim.push_back(FLAGS_tensorrt_engine_batch_size);
      for (int i = 0; i < dims.nbDims; i++) {
        ddim.push_back(dims.d[i]);
      }

      auto *fluid_v = scope.FindVar(y);
      PADDLE_ENFORCE_NOT_NULL(fluid_v, "no output variable called %s", y);
      auto *fluid_t = fluid_v->GetMutable<framework::LoDTensor>();

      fluid_t->Resize(framework::make_ddim(ddim));

      // TODO(Superjomn) find some way to determine which device to output the
      // tensor.
      // if (platform::is_cpu_place(fluid_t->place())) {
      // TODO(Superjomn) change this float to dtype size.
      auto size = inference::analysis::AccuDims(dims.d, dims.nbDims) *
                  FLAGS_tensorrt_engine_batch_size;
      engine->GetOutputInGPU(
          output_maps[output_index],
          fluid_t->mutable_data<float>(platform::CUDAPlace(
              boost::get<platform::CUDAPlace>(dev_place).device)),
          size * sizeof(float));

      output_index += 1;
    }

    cudaStreamSynchronize(*engine->stream());
  }

  void Prepare(const framework::Scope &scope, const platform::Place &dev_place,
               std::string precision_mode,
               TRTInt8Calibrator *calibrator) const {
    framework::proto::BlockDesc block_desc;
    block_desc.ParseFromString(Attr<std::string>("subgraph"));
    VLOG(4) << "Prepare engine";
    // Get the ProgramDesc and pass to convert.

    std::vector<std::string> output_maps =
        Attr<std::vector<std::string>>("output_name_mapping");

    // TODO(Superjomn) replace this with a different stream
    auto *engine = Singleton<TRT_EngineManager>::Global().Create(
        max_batch_size_, workspace_size_,
        nullptr /*engine hold its own stream*/,
        Attr<std::string>("engine_uniq_key"),
        boost::get<platform::CUDAPlace>(dev_place).device, precision_mode,
        calibrator);

    engine->InitNetwork();

    framework::BlockDesc block(nullptr /*programdesc*/, &block_desc);
    VLOG(4) << "parsed var size " << block.AllVars().size();
    // Add inputs
    VLOG(4) << "declare inputs";
    for (auto &input : Inputs("Xs")) {
      if (param_names_.count(input)) continue;
      VLOG(4) << "declare input " << input;
      auto *var = block.FindVar(input);
      // TensorRT engine need to create parameters. The parameter's description
      // should be set in
      PADDLE_ENFORCE(var, "no variable called %s", input);
      PADDLE_ENFORCE_EQ(var->GetType(), FluidDT::VarType_Type_LOD_TENSOR,
                        "TensorRT engine only takes LoDTensor as input");
      auto shape = var->GetShape();
      // For the special batch_size placeholder -1, drop it and pass the real
      // shape of data.
      // TODO(Superjomn) fix this with batch broadcast, or it can't handle
      // variational batch size.
      if (shape[0] == -1) {
        shape[0] = FLAGS_tensorrt_engine_batch_size;
      }
      engine->DeclareInput(
          input, FluidDataType2TRT(
                     var->Proto()->type().lod_tensor().tensor().data_type()),
          Vec2TRT_Dims(shape));
    }

    inference::Singleton<inference::tensorrt::OpConverter>::Global()
        .ConvertBlock(block_desc, param_names_, scope, engine);

    // Add outputs
    for (auto &output : output_maps) {
      engine->DeclareOutput(output);
    }
    engine->FreezeNetwork();
  }
};

}  // namespace operators
}  // namespace paddle

#endif  // PADDLE_WITH_CUDA
