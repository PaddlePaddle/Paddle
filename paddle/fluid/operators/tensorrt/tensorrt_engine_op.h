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

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"

namespace paddle {

namespace operators {

using inference::Singleton;
using inference::tensorrt::TensorRTEngine;
using inference::tensorrt::TRTInt8Calibrator;
using inference::tensorrt::TRTCalibratorEngine;
using inference::tensorrt::TRTCalibratorEngineManager;

class TensorRTEngineOp : public framework::OperatorBase {
 private:
  std::vector<std::string> input_names_;
  std::unordered_set<std::string> param_names_;
  mutable std::unique_ptr<TensorRTEngine> trt_engine_;
  int max_batch_size_;
  int workspace_size_;
  std::unique_ptr<TRTInt8Calibrator> calibrator_;
  bool enable_int8_;
  std::string calibration_data_;
  std::string engine_key_;
  std::string engine_serialized_data_;
  bool calibration_mode_;

 public:
  TensorRTEngineOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {
    input_names_ = Inputs("Xs");
    max_batch_size_ = Attr<int>("max_batch_size");
    workspace_size_ = Attr<int>("workspace_size");
    enable_int8_ = Attr<bool>("enable_int8");
    calibration_data_ = Attr<std::string>("calibration_data");
    engine_key_ = Attr<std::string>("engine_key");
    engine_serialized_data_ = Attr<std::string>("engine_serialized_data");

    auto params = Attr<std::vector<std::string>>("parameters");
    for (const auto &param : params) {
      param_names_.insert(param);
    }
    // calibration_mode is ture represents we need to
    // generate the calibration table data.
    calibration_mode_ = (enable_int8_ && calibration_data_.size() == 0);

    VLOG(4) << "calibration_mode: " << calibration_mode_;
    if (enable_int8_ && calibration_data_.size()) {
      calibrator_.reset(new TRTInt8Calibrator(calibration_data_));
    }
  }

 protected:
  void RunNativeImpl(const framework::Scope &scope,
                     const platform::Place &dev_place) const {
    framework::Executor executor(dev_place);
    auto *block = Attr<framework::BlockDesc *>("sub_block");
    auto *program = block->Program();
    auto &current_scope = scope.NewScope();
    auto ctx = executor.Prepare(*program, block->ID());
    executor.RunPreparedContext(ctx.get(), &current_scope, false, true, true);
  }

  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    if (calibration_mode_ == true) {
      RunCalibration(scope, dev_place);
      return;
    }
    auto *trt_engine = GetEngine(scope, dev_place);
    RunTrt(scope, dev_place, trt_engine);
  }

  void RunCalibration(const framework::Scope &scope,
                      const platform::Place &dev_place) const {
    // This process will builds a 32-bit trt engine, runs it on the calibration
    // set, and records a histogram for each
    // tensor of the distribution of activation values.
    LOG_FIRST_N(INFO, 1) << "The TRT engine: " << engine_key_
                         << " is running calibration trt int8... ";
    int runtime_batch = 1;
    if (!Singleton<TRTCalibratorEngineManager>::Global().Has(engine_key_)) {
      TRTCalibratorEngine *calib_res =
          Singleton<TRTCalibratorEngineManager>::Global().Create(engine_key_);
      std::unordered_map<std::string, size_t> calib_buffers;
      for (auto &x : input_names_) {
        if (param_names_.count(x)) continue;
        auto &t =
            inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);
        calib_buffers[x] = t.memory_size();
        auto t_shape = framework::vectorize(t.dims());
        runtime_batch = t_shape[0];
      }
      calib_res->calib_.reset(new TRTInt8Calibrator(
          calib_buffers, runtime_batch, engine_key_, dev_place));
      calib_res->thr_.reset(new std::thread([&]() {
        calib_res->engine_.reset(new TensorRTEngine(
            max_batch_size_, workspace_size_, enable_int8_,
            calib_res->calib_.get(),
            boost::get<platform::CUDAPlace>(dev_place).device));
        VLOG(3) << "start the calib trt engine thread";
        PrepareTRTEngine(scope, calib_res->engine_.get());
      }));
    }

    TRTInt8Calibrator *temp_calibrator =
        Singleton<TRTCalibratorEngineManager>::Global()
            .Get(engine_key_)
            ->calib_.get();
    std::unordered_map<std::string, void *> calib_data;

    for (auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      auto &t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);
      calib_data.emplace(x, t.data<void>());
    }
    temp_calibrator->setBatch(calib_data);
    RunNativeImpl(scope, dev_place);
  }

  void RunTrt(const framework::Scope &scope, const platform::Place &dev_place,
              TensorRTEngine *engine) const {
    int runtime_batch = 1;
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext &>(dev_ctx).stream();

    PADDLE_ENFORCE(!input_names_.empty(), "should pass more than one inputs");

    std::vector<std::string> output_maps =
        Attr<std::vector<std::string>>("output_name_mapping");

    int num_inputs = 0;

    for (const auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      num_inputs += 1;
    }
    const int num_bindings = num_inputs + Outputs("Ys").size();
    std::vector<void *> buffers(num_bindings);

    // Bind input tensor to TRT.
    for (const auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      // convert input and copy to TRT engine's buffer
      auto &t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);
      auto t_shape = framework::vectorize(t.dims());
      runtime_batch = t_shape[0];

      const int bind_index = engine->engine()->getBindingIndex(x.c_str());
      PADDLE_ENFORCE(bind_index < num_bindings,
                     "The bind index should be less than num_bindings");
      buffers[bind_index] = static_cast<void *>(t.data<float>());
    }

    // Bind output tensor to TRT.
    int output_index = 0;
    VLOG(4) << "TensorRT Engine Op Outputs:";
    for (const auto &y : Outputs("Ys")) {
      const int bind_index =
          engine->engine()->getBindingIndex(output_maps[output_index].c_str());
      auto dims = engine->engine()->getBindingDimensions(bind_index);
      // Use the output ITensor's dims to reshape the Fluid Tensor.
      // The ITensor doesn't contain the batch size dim.
      std::vector<int> ddim;
      ddim.push_back(runtime_batch);
      for (int i = 0; i < dims.nbDims; i++) {
        ddim.push_back(dims.d[i]);
      }
      auto *fluid_v = scope.FindVar(y);
      PADDLE_ENFORCE_NOT_NULL(fluid_v, "no output variable called %s", y);
      auto *fluid_t = fluid_v->GetMutable<framework::LoDTensor>();
      fluid_t->Resize(framework::make_ddim(ddim));

      PADDLE_ENFORCE(bind_index < num_bindings,
                     "The bind index should be less than num_bindings");
      buffers[bind_index] = static_cast<void *>(fluid_t->mutable_data<float>(
          boost::get<platform::CUDAPlace>(dev_place)));

      output_index += 1;
    }

    PADDLE_ENFORCE_LE(runtime_batch, max_batch_size_);
    // Execute the engine.
    engine->Execute(runtime_batch, &buffers, stream);
    cudaStreamSynchronize(stream);
  }

  TensorRTEngine *GetEngine(const framework::Scope &scope,
                            const platform::Place &dev_place) const {
    if (!trt_engine_) {
      trt_engine_.reset(new inference::tensorrt::TensorRTEngine(
          max_batch_size_, workspace_size_, enable_int8_, calibrator_.get(),
          boost::get<platform::CUDAPlace>(dev_place).device));
      if (!engine_serialized_data_.empty()) {
        trt_engine_->Deserialize(engine_serialized_data_);
      } else {
        PrepareTRTEngine(scope, trt_engine_.get());
      }
    }
    return trt_engine_.get();
  }

  void PrepareTRTEngine(const framework::Scope &scope,
                        TensorRTEngine *engine) const {
    LOG(INFO) << "Prepare TRT engine (Optimize model structure, Select OP "
                 "kernel etc). This process may cost a lot of time.";
    framework::proto::BlockDesc block_proto;
    block_proto.ParseFromString(Attr<std::string>("subgraph"));
    framework::BlockDesc block_desc(nullptr, &block_proto);

    std::vector<std::string> inputs = Inputs("Xs");
    std::vector<std::string> outputs =
        Attr<std::vector<std::string>>("output_name_mapping");

    inference::Singleton<inference::tensorrt::OpConverter>::Global()
        .ConvertBlockToTRTEngine(&block_desc, scope, inputs, param_names_,
                                 outputs, engine);
  }
};

}  // namespace operators
}  // namespace paddle

#endif  // PADDLE_WITH_CUDA
