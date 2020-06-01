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
#include <utility>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/helper.h"

namespace paddle {

namespace operators {

using inference::Singleton;
using inference::tensorrt::TensorRTEngine;
using inference::tensorrt::TRTInt8Calibrator;
using inference::tensorrt::TRTCalibratorEngine;
using inference::tensorrt::TRTCalibratorEngineManager;

static void RuntimeStaticShapeCheck(std::vector<int64_t> runtime_input_shape,
                                    std::vector<int64_t> model_input_shape) {
  auto comma_fold = [](std::string a, int b) {
    return std::move(a) + ", " + std::to_string(b);
  };
  std::string model_input_shape_str = std::accumulate(
      std::next(model_input_shape.begin()), model_input_shape.end(),
      std::to_string(model_input_shape[0]), comma_fold);
  std::string runtime_input_shape_str = std::accumulate(
      std::next(runtime_input_shape.begin()), runtime_input_shape.end(),
      std::to_string(runtime_input_shape[0]), comma_fold);
  PADDLE_ENFORCE_EQ(
      model_input_shape == runtime_input_shape, true,
      platform::errors::InvalidArgument(
          "Input shapes are inconsistent with the model. Expect [%s] in "
          "model description, but got [%s] in runtime. TRT 5 "
          "or lower version "
          "does not support dynamic input shapes. Please check and "
          "modify "
          "your input shapes.",
          model_input_shape_str, runtime_input_shape_str));
}

class TensorRTEngineOp : public framework::OperatorBase {
 private:
  std::vector<std::string> input_names_;
  std::unordered_set<std::string> param_names_;
  mutable TensorRTEngine *trt_engine_{nullptr};
  int max_batch_size_;
  int workspace_size_;
  std::unique_ptr<TRTInt8Calibrator> calibrator_;
  bool enable_int8_;
  bool enable_fp16_;
  bool use_calib_mode_;
  std::string calibration_data_;
  std::string engine_key_;
  bool calibration_mode_;
  int predictor_id_;
  int device_id_;
  AnalysisConfig::Precision precision_mode_;

 public:
  TensorRTEngineOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {
    input_names_ = Inputs("Xs");
    max_batch_size_ = Attr<int>("max_batch_size");
    workspace_size_ = Attr<int>("workspace_size");
    device_id_ = Attr<int>("gpu_id");
    enable_int8_ = Attr<bool>("enable_int8");
    enable_fp16_ = Attr<bool>("enable_fp16");
    use_calib_mode_ = Attr<bool>("use_calib_mode");
    calibration_data_ = Attr<std::string>("calibration_data");
    engine_key_ = Attr<std::string>("engine_key");
    predictor_id_ = Attr<int>("predictor_id");

    auto params = Attr<std::vector<std::string>>("parameters");
    for (const auto &param : params) {
      param_names_.insert(param);
    }
    // calibration_mode is ture represents we need to
    // generate the calibration table data.
    calibration_mode_ =
        (enable_int8_ && calibration_data_.size() == 0 && use_calib_mode_);

    VLOG(4) << "calibration_mode: " << calibration_mode_;
    if (enable_int8_ && calibration_data_.size()) {
      calibrator_.reset(new TRTInt8Calibrator(calibration_data_));
    }
    bool has_engine =
        inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
            .Has(engine_key_ + std::to_string(predictor_id_));

    if (!calibration_mode_ && has_engine) {
      trt_engine_ =
          inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
              .Get(engine_key_ + std::to_string(predictor_id_));
    }
    precision_mode_ = AnalysisConfig::Precision::kFloat32;
    if (enable_int8_) {
      precision_mode_ = AnalysisConfig::Precision::kInt8;
    }
    if (enable_fp16_) {
      precision_mode_ = AnalysisConfig::Precision::kHalf;
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
    LOG_FIRST_N(INFO, 1) << "This process is generating calibration table for "
                            "Paddle TRT int8...";

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
            max_batch_size_, workspace_size_, precision_mode_,
            calib_res->calib_.get(),
            BOOST_GET_CONST(platform::CUDAPlace, dev_place).device));
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

    PADDLE_ENFORCE_EQ(input_names_.empty(), false,
                      "should pass at least one input");

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
      auto t_shape = framework::vectorize<int64_t>(t.dims());
      runtime_batch = t_shape[0];
      const int bind_index = engine->engine()->getBindingIndex(x.c_str());
      PADDLE_ENFORCE_LT(
          bind_index, num_bindings,
          platform::errors::InvalidArgument(
              "Wrong TRT engine input binding index. Expected The "
              "binding index of TRT engine input to be less than "
              "the number of inputs and outputs. Received binding "
              "index=%d >= total inputs and outputs=%d",
              bind_index, num_bindings));
      if (!engine->with_dynamic_shape()) {
        // check if the input shapes are consistent with model.
        if (HasAttr(x + "_shape")) {
          std::vector<int64_t> i_shape =
              Attr<std::vector<int64_t>>(x + "_shape");
          std::vector<int64_t> model_input_shape(i_shape.begin() + 1,
                                                 i_shape.end());
          std::vector<int64_t> runtime_input_shape(t_shape.begin() + 1,
                                                   t_shape.end());
          RuntimeStaticShapeCheck(runtime_input_shape, model_input_shape);
        }
      } else {
#if IS_TRT_VERSION_GE(6000)
        auto *trt_context = engine->context();
        trt_context->setBindingDimensions(
            bind_index, inference::tensorrt::Vec2TRT_Dims(t_shape, x, true));
#endif
      }
      auto type = t.type();
      if (type == framework::proto::VarType::FP32) {
        buffers[bind_index] = static_cast<void *>(t.data<float>());
      } else if (type == framework::proto::VarType::INT64) {
        buffers[bind_index] = static_cast<void *>(t.data<int64_t>());
      } else {
        PADDLE_THROW(platform::errors::Fatal(
            "The TRT Engine OP only support float and int64_t input."));
      }
    }

    // Bind output tensor to TRT.
    int output_index = 0;
    VLOG(4) << "TensorRT Engine Op Outputs:";
    for (const auto &y : Outputs("Ys")) {
      const int bind_index =
          engine->engine()->getBindingIndex(output_maps[output_index].c_str());
      std::vector<int> ddim;

      if (!engine->with_dynamic_shape()) {
        auto dims = engine->engine()->getBindingDimensions(bind_index);
        ddim.push_back(runtime_batch);
        for (int i = 0; i < dims.nbDims; i++) {
          ddim.push_back(dims.d[i]);
        }
      } else {
#if IS_TRT_VERSION_GE(6000)
        auto *trt_context = engine->context();
        auto dims = trt_context->getBindingDimensions(bind_index);
        int nb_dims = dims.nbDims;
        for (; nb_dims > 0; nb_dims--) {
          if (dims.d[nb_dims - 1] != 1) break;
        }
        for (int i = 0; i < nb_dims; i++) ddim.push_back(dims.d[i]);
#endif
      }
      auto *fluid_v = scope.FindVar(y);
      PADDLE_ENFORCE_NOT_NULL(fluid_v, "no output variable called %s", y);
      auto *fluid_t = fluid_v->GetMutable<framework::LoDTensor>();
      fluid_t->Resize(framework::make_ddim(ddim));

      PADDLE_ENFORCE(bind_index < num_bindings,
                     "The bind index should be less than num_bindings");
      buffers[bind_index] = static_cast<void *>(fluid_t->mutable_data<float>(
          BOOST_GET_CONST(platform::CUDAPlace, dev_place)));

      output_index += 1;
    }

    if (!engine->with_dynamic_shape()) {
      PADDLE_ENFORCE_LE(
          runtime_batch, max_batch_size_,
          platform::errors::InvalidArgument(
              "The runtime batch size (%d) is greater than the max batch "
              "size(%d).\n"
              "There are two possible causes for this problem: \n"
              "1. Check whether the runtime batch is larger than the max_batch "
              "set by EnableTensorrtEngine()\n"
              "2. Check whether the model you are running has multiple trt "
              "subgraphs: \n "
              "\tIf there are multiple trt subgraphs, you need to ensure that "
              "the first dimension of the input tensor of these subgraphs is "
              "consistent.\n"
              "\tIf there are inconsistent subgraphs, you need to filter them "
              "by "
              "setting min_subgraph_size using EnableTensorrtEngine "
              "interface.\n"
              "\tThe min_subgraph_size shouble to be greater than the number "
              "of "
              "nodes in the inconsistent subgraph.\n",
              runtime_batch, max_batch_size_));
    }
    // Execute the engine.
    engine->Execute(runtime_batch, &buffers, stream);
  }

  TensorRTEngine *GetEngine(const framework::Scope &scope,
                            const platform::Place &dev_place) const {
    if (!trt_engine_) {
      trt_engine_ =
          inference::Singleton<inference::tensorrt::TRTEngineManager>::Global()
              .Create(engine_key_ + std::to_string(predictor_id_),
                      max_batch_size_, workspace_size_, precision_mode_,
                      calibrator_.get(), device_id_);
      PrepareTRTEngine(scope, trt_engine_);
    }
    return trt_engine_;
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
