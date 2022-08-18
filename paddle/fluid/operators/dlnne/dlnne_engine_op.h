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
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_device_transform.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/utils/io_utils.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/phi/core/ddim.h"

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"

namespace dl {
namespace nne {
class Builder;
class Engine;
class Network;
class Parser;
class ExecutionContext;

inline unsigned int GetElementSize(DataType type) {
  switch (type) {
    case DataType::kINT64:
    case DataType::kUINT64:
    case DataType::kFLOAT64:
      return 8;
    case DataType::kINT32:
    case DataType::kUINT32:
    case DataType::kFLOAT32:
      return 4;
    case DataType::kINT16:
    case DataType::kUINT16:
    case DataType::kFLOAT16:
      return 2;
    case DataType::kINT8:
    case DataType::kUINT8:
    case DataType::kBOOL:
      return 1;
    case DataType::kUNKNOWN_TYPE:
      return 0;
  }
  return 0;
}

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

std::string ConvertType(paddle::experimental::DataType type);

int GetDataByte(paddle::experimental::DataType type);

std::string GenerateRandomKey();

void ConvertPaddle2Onnx(std::string onnx_file_name,
                        std::string subgraph_root_path);

void QuantizeOnnx(std::string onnx_file_name,
                  std::string rlym_file_name,
                  std::string quantized_rlym_file_name,
                  std::string dataset_path,
                  std::string dataset_plugin_path);

static paddle::experimental::DataType DLNNE2FluidDataType(
    dl::nne::DataType type) {
  switch (type) {
    case dl::nne::DataType::kFLOAT32:
      return paddle::experimental::DataType::FLOAT32;
    case dl::nne::DataType::kINT32:
      return paddle::experimental::DataType::INT32;
    case dl::nne::DataType::kINT64:
      return paddle::experimental::DataType::INT64;
    case dl::nne::DataType::kFLOAT16:
      return paddle::experimental::DataType::FLOAT16;
    case dl::nne::DataType::kUINT8:
      return paddle::experimental::DataType::UINT8;
    case dl::nne::DataType::kINT8:
      return paddle::experimental::DataType::INT8;
    case dl::nne::DataType::kBOOL:
      return paddle::experimental::DataType::BOOL;
    default:
      PADDLE_THROW(platform::errors::InvalidArgument(
          "unknown fluid datatype in Fluid op converter"));
      return paddle::experimental::DataType::FLOAT32;
  }
}

}  // namespace inference
}  // namespace paddle

namespace paddle {

namespace operators {

std::mutex static dlnne_create_lock;

class DlnneEngineOp : public framework::OperatorBase {
 private:
  std::vector<std::string> input_names_;
  std::unordered_set<std::string> param_names_;
  std::string engine_key_;
  bool use_static_batch_;
  bool calibration_mode_;
  std::string calibration_data_path_;
  std::string subgraph_root_path_;
  bool enable_int8_;
  bool use_calib_mode_;

  std::string weight_share_mode_;
  int max_batch_size_;
  int num_inputs;
  int num_outputs;
  // std::vector<std::string> output_names;
  // std::vector<std::string> input_names;

  dl::nne::Builder *builder;
  dl::nne::Parser *parser;
  dl::nne::Network *network;
  dl::nne::ExecutionContext *context;
  dl::nne::Engine *engine;

  unsigned int engine_input_size;
  std::vector<int> InputIndexToBindIndex_;

  char *dump_flag_;
  char *dlnne_log_flag_;
  char *dl_sdk_dir_;

 public:
  DlnneEngineOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {
    input_names_ = Inputs("Xs");
    engine_key_ = Attr<std::string>("engine_key");
    use_static_batch_ = Attr<bool>("use_static_batch");
    max_batch_size_ = Attr<int32_t>("max_batch_size");
    weight_share_mode_ = Attr<std::string>("weight_share_mode");
    calibration_mode_ = Attr<bool>("calibration_mode");
    calibration_data_path_ = Attr<std::string>("calibration_data_path");
    subgraph_root_path_ = Attr<std::string>("subgraph_root_path");
    enable_int8_ = Attr<bool>("enable_int8");
    use_calib_mode_ = Attr<bool>("use_calib_mode");

    // dump input/output buffer of dlnne engine
    dump_flag_ = getenv("PADDLE_DUMP_DLNNE_BUFFER");
    dlnne_log_flag_ = getenv("PADDLE_DLNNE_LOG");
    dl_sdk_dir_ = getenv("DL_SDK_DIR");

    auto params = Attr<std::vector<std::string>>("parameters");
    for (const auto &param : params) {
      param_names_.insert(param);
    }

    std::vector<std::string> XsMap;
    num_inputs = Inputs("Xs").size();
    std::string valid_input_name_str = Attr<std::string>("valid_input_names");

    for (const auto &x : Inputs("Xs")) {
      // input_names.push_back(x);
      XsMap.push_back(
          valid_input_name_str.substr(0, valid_input_name_str.find(",")));
      valid_input_name_str =
          valid_input_name_str.substr(valid_input_name_str.find(",") + 1);
    }
    std::vector<std::string> YsMap;

    num_outputs = Outputs("Ys").size();
    std::string valid_output_name_str = Attr<std::string>("valid_output_names");
    for (const auto &y : Outputs("Ys")) {
      // output_names.push_back(y);
      YsMap.push_back(
          valid_output_name_str.substr(0, valid_output_name_str.find(",")));
      valid_output_name_str =
          valid_output_name_str.substr(valid_output_name_str.find(",") + 1);
    }

    // TODO(pei.jiang): add dlnne_engine manager to manage dlnne_engine
    if (!calibration_mode_) {
      std::map<std::string, dl::nne::WeightShareMode> weight_share_map;
      weight_share_map.insert(
          std::make_pair("0", dl::nne::WeightShareMode::kSingle));
      weight_share_map.insert(
          std::make_pair("1", dl::nne::WeightShareMode::kSingle));
      weight_share_map.insert(
          std::make_pair("2", dl::nne::WeightShareMode::kSingle));
      weight_share_map.insert(
          std::make_pair("3", dl::nne::WeightShareMode::kSingle));
      weight_share_map.insert(
          std::make_pair("01", dl::nne::WeightShareMode::kShare2));
      weight_share_map.insert(
          std::make_pair("23", dl::nne::WeightShareMode::kShare2));
      weight_share_map.insert(
          std::make_pair("0123", dl::nne::WeightShareMode::kShare4));

      std::map<std::string, dl::nne::ClusterConfig> cluster_config_map;
      cluster_config_map.insert(
          std::make_pair("0", dl::nne::ClusterConfig::kCluster0));
      cluster_config_map.insert(
          std::make_pair("1", dl::nne::ClusterConfig::kCluster1));
      cluster_config_map.insert(
          std::make_pair("2", dl::nne::ClusterConfig::kCluster2));
      cluster_config_map.insert(
          std::make_pair("3", dl::nne::ClusterConfig::kCluster3));
      cluster_config_map.insert(
          std::make_pair("01", dl::nne::ClusterConfig::kCluster01));
      cluster_config_map.insert(
          std::make_pair("23", dl::nne::ClusterConfig::kCluster23));
      cluster_config_map.insert(
          std::make_pair("0123", dl::nne::ClusterConfig::kCluster0123));

      dl::nne::WeightShareMode mode = weight_share_map[weight_share_mode_];
      dl::nne::ClusterConfig cluster_config =
          cluster_config_map[weight_share_mode_];
      if (dlnne_log_flag_) {
        LOG(INFO) << "weight_share_mode: " << mode
                  << " cluster_config: " << cluster_config;
      }

      std::string onnx_file_name =
          subgraph_root_path_ + "/" + engine_key_ + ".onnx";
      inference::ConvertPaddle2Onnx(onnx_file_name, subgraph_root_path_);

      std::string rlym_file_name =
          subgraph_root_path_ + "/" + engine_key_ + ".rlym";
      // quantize don't support set quantized ouput model path now,
      // the quantized model file is in current dir
      std::string quantized_rlym_file_name = engine_key_ + ".quantized.rlym";

      std::stringstream filename;
      std::stringstream engine_file_name;

      if (enable_int8_ && use_calib_mode_) {
        std::string dataset_path = calibration_data_path_;
        std::string cnt_dataset_path = dataset_path + "/" + input_names_[0];

        std::stringstream dataset_plugin_path;
        dataset_plugin_path << dl_sdk_dir_
                            << "/python/dleol/quantize/plugin.py";

        inference::QuantizeOnnx(onnx_file_name,
                                rlym_file_name,
                                quantized_rlym_file_name,
                                dataset_path,
                                dataset_plugin_path.str());

        filename << quantized_rlym_file_name;
        engine_file_name << subgraph_root_path_ << "/" << engine_key_
                         << "_quantized"
                         << "_ws_" << weight_share_mode_ << ".engine";
      } else {
        filename << onnx_file_name;
        engine_file_name << subgraph_root_path_ << "/" << engine_key_ << "_ws_"
                         << weight_share_mode_ << ".engine";
      }

      dlnne_create_lock.lock();
      if (dlnne_log_flag_) {
        LOG(INFO) << "EngineKey:" << engine_key_
                  << " use_static_batch_:" << use_static_batch_
                  << " max_batch_size_:" << max_batch_size_
                  << " weight_share_mode_: " << weight_share_mode_;
      }

      builder = dl::nne::CreateInferBuilder();
      PADDLE_ENFORCE_NE(
          builder,
          nullptr,
          platform::errors::Unavailable("nne create builder failed"));
      dl::nne::BuilderConfig builder_cfg;
      builder_cfg.max_batch_size = max_batch_size_;
      builder_cfg.ws_mode = weight_share_map[weight_share_mode_];
      builder->SetBuilderConfig(builder_cfg);
      network = builder->CreateNetwork();

      parser = dl::nne::CreateParser();
      PADDLE_ENFORCE_NE(
          parser,
          nullptr,
          platform::errors::Unavailable("nne create parser failed"));
      if (dlnne_log_flag_) {
        LOG(INFO) << "set output for dlnne";
      }
      for (std::string &output_op_name : YsMap) {
        parser->RegisterOutput(output_op_name.c_str());
        if (dlnne_log_flag_) {
          LOG(INFO) << output_op_name;
        }
      }

      std::fstream engine_file;
      engine_file.open(engine_file_name.str().c_str(), std::ios::in);
      if (!engine_file) {
        if (dlnne_log_flag_) {
          LOG(INFO) << "parser model file for dlnne";
        }
        parser->Parse(filename.str().c_str(), *network);
        if (dlnne_log_flag_) {
          LOG(INFO) << "build network";
        }
        engine = builder->BuildEngine(*network);

        auto memory = engine->Serialize();
        std::ofstream out(engine_file_name.str().c_str(),
                          std::ofstream::binary);
        out.write(reinterpret_cast<char *>(memory->Data()), memory->Size());
        out.close();
        memory->Destroy();
      } else {
        engine_file.seekg(0, std::ios::end);
        uint64_t length = static_cast<uint64_t>(engine_file.tellg());
        engine_file.seekg(0, std::ios::beg);
        char *slz_data = new char[length];
        engine_file.read(slz_data, static_cast<int64_t>(length));
        engine = dl::nne::Deserialize(slz_data, length);
        delete[] slz_data;
      }

      engine_input_size = num_inputs + num_outputs;
      for (std::string &input_name : XsMap) {
        int BindIndex = engine->GetBindingIndex(input_name.c_str());
        InputIndexToBindIndex_.push_back(BindIndex);
      }
      for (std::string &output_name : YsMap) {
        int BindIndex = engine->GetBindingIndex(output_name.c_str());
        InputIndexToBindIndex_.push_back(BindIndex);
      }

      // context
      context = engine->CreateExecutionContext(
          cluster_config_map[weight_share_mode_]);
      dlnne_create_lock.unlock();
    }
  }

  ~DlnneEngineOp() {
    if (!calibration_mode_) {
      network->Destroy();
      context->Destroy();
      engine->Destroy();
      parser->Destroy();
      builder->Destroy();
    }
  }

 protected:
  void RunDlnneOnCreateEngine(const framework::Scope &scope,
                              const platform::Place &dev_place) const {
    PADDLE_ENFORCE_EQ(
        input_names_.empty(),
        false,
        platform::errors::PreconditionNotMet(
            "Dlnne engine needs at least one input, but no input is found. "
            "Please check if you set the input correctly."));

    std::vector<void *> input_buffers(num_inputs);
    std::vector<void *> cpu_input_buffers(num_inputs);
    std::vector<std::vector<int64_t>> input_shapes(num_inputs);
    std::vector<int32_t> input_data_types(num_inputs);
    std::vector<int64_t> input_bytes(num_inputs);

    dlnne_create_lock.lock();
    int index = 0;
    int infer_batch = 1;
    std::vector<int> vec_infer_batch;
    // compute infer_batch
    if (use_static_batch_) {
      for (const auto &x : Inputs("Xs")) {
        if (param_names_.count(x)) continue;
        // convert input and copy to Dlnne engine's buffer
        auto &t =
            inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);

        auto t_shape = phi::vectorize<int64_t>(t.dims());
        std::vector<int64_t> runtime_input_shape(t_shape.begin(),
                                                 t_shape.end());
        const int bind_index = index;
        index++;
        dl::nne::Dims in_dim = engine->GetBindingDimensions(bind_index);

        int compute_batch = runtime_input_shape[0] / in_dim.d[0];
        VLOG(4) << "compute batch: " << compute_batch;
        vec_infer_batch.push_back(compute_batch);
      }

      int first_batch = vec_infer_batch[0];
      for (auto batch : vec_infer_batch) {
        PADDLE_ENFORCE_EQ(
            first_batch,
            batch,
            platform::errors::Unavailable(
                "compute infer_batchs is different from each other"));
      }
      infer_batch = first_batch;
    }

    index = 0;
    for (const auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      // convert input and copy to Dlnne engine's buffer
      auto &t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);

      const int bind_index = index;
      index++;
      int64_t data_bytes, ele_num;
      int32_t dtype;
      auto type = t.type();
      data_bytes = 1;
      ele_num = 1;
      void *buffer = nullptr;
      // TODO(pei.jiang): add more type
      if (type == paddle::experimental::DataType::FLOAT32) {
        buffer = static_cast<void *>(t.data<float>());
        data_bytes = 4;
        dtype = 0;
      } else if (type == paddle::experimental::DataType::INT64) {
        buffer = static_cast<void *>(t.data<int64_t>());
        data_bytes = 8;
        dtype = 1;
      } else if (type == paddle::experimental::DataType::INT32) {
        buffer = static_cast<void *>(t.data<int32_t>());
        data_bytes = 4;
        dtype = 2;
      } else if (type == paddle::experimental::DataType::FLOAT16) {
        buffer = static_cast<void *>(t.data<paddle::platform::float16>());
        data_bytes = 2;
        dtype = 3;
      } else {
        PADDLE_THROW(
            platform::errors::Fatal("The DLNNE Engine OP only support "
                                    "float/int32_t/int64_t/float16 input."));
      }
      input_buffers[bind_index] = buffer;

      auto t_shape = phi::vectorize<int64_t>(t.dims());
      std::vector<int64_t> runtime_input_shape(t_shape.begin(), t_shape.end());
      for (auto &size : t_shape) {
        data_bytes = data_bytes * size;
        ele_num = ele_num * size;
      }

      VLOG(4) << "buffers_size:" << data_bytes;
      cpu_input_buffers[bind_index] =
          input_buffers[bind_index];  // malloc(data_bytes);
      input_shapes[bind_index] = runtime_input_shape;
      input_data_types[bind_index] = dtype;
      input_bytes[bind_index] = data_bytes;

      if (dump_flag_) {
        std::stringstream dump_input_name;
        dump_input_name << engine_key_ << "_input_" << bind_index << ".txt";
        std::ofstream dump_input_file;
        dump_input_file.open(dump_input_name.str());
        for (int64_t i = 0; i < ele_num; i++) {
          dump_input_file << static_cast<float *>(
                                 cpu_input_buffers[bind_index])[i]
                          << "\n";
        }
        dump_input_file << "\b";
        dump_input_file.close();
      }
    }

    // output shape
    std::vector<std::vector<int64_t>> out_shapes;
    std::vector<dl::nne::DataType> out_types;
    std::vector<int64_t> out_ele_nums;
    std::vector<int32_t> output_bytes;
    for (int i = 0; i < num_outputs; i++) {
      int index = InputIndexToBindIndex_[i + num_inputs];
      dl::nne::DataType out_type = engine->GetBindingDataType(index);
      out_types.push_back(out_type);
      dl::nne::Dims out_dim = engine->GetBindingDimensions(index);
      std::vector<int64_t> shape(out_dim.nbDims);
      for (int dim = 0; dim < out_dim.nbDims; dim++) {
        if (use_static_batch_ && dim == 0) {
          shape[dim] = (out_dim.d[dim]) * infer_batch;
        } else {
          shape[dim] = (out_dim.d[dim]);
        }
      }

      out_shapes.push_back(shape);
      int64_t data_bytes, out_ele_num;
      out_ele_num = 1;

      // float32
      data_bytes = dl::nne::GetElementSize(out_type);
      for (auto &size : shape) {
        data_bytes = data_bytes * size;
        out_ele_num = out_ele_num * size;
      }
      VLOG(4) << "data_bytes: " << data_bytes;
      output_bytes.push_back(data_bytes);
      out_ele_nums.push_back(out_ele_num);
    }

    int bind_index = 0;
    std::vector<void *> cpu_output_buffers(num_outputs);
    std::vector<void *> output_buffers(num_outputs);

    for (const auto &y : Outputs("Ys")) {
      auto *fluid_v = scope.FindVar(y);
      PADDLE_ENFORCE_NOT_NULL(
          fluid_v,
          platform::errors::NotFound(
              "Output variable %s is not found in DLNNE subgraph.", y));

      auto *fluid_t = fluid_v->GetMutable<framework::LoDTensor>();

      VLOG(4) << bind_index << ": out_shapes[bind_index] dim:"
              << out_shapes[bind_index].size();
      fluid_t->Resize(phi::make_ddim(out_shapes[bind_index]));

      dl::nne::DataType dl_type = out_types[bind_index];
      if (dlnne_log_flag_) {
        LOG(INFO) << "output type: " << dl_type;
      }
      output_buffers[bind_index] = static_cast<void *>(fluid_t->mutable_data(
          dev_place, inference::DLNNE2FluidDataType(dl_type)));

      cpu_output_buffers[bind_index] =
          output_buffers[bind_index];  // malloc(data_bytes);
      bind_index++;
    }

    std::vector<void *> engine_input_ptr(engine_input_size);

    // set input_ptr
    for (unsigned int i = 0; i < engine_input_size; i++) {
      if (InputIndexToBindIndex_[i] < 0) {
        continue;
      }

      if (engine->BindingIsInput(InputIndexToBindIndex_[i])) {
        // copy cpu buffer to gpu buffer
        int64_t total_bytes;
        total_bytes = input_bytes[i];
        VLOG(4) << "input_bytes: " << total_bytes;

        void *gpu_ptr;
        cudaMalloc(&gpu_ptr, total_bytes);
        engine_input_ptr[InputIndexToBindIndex_[i]] = gpu_ptr;

        paddle::inference::CopyTensorCpuToDevice(
            gpu_ptr,
            reinterpret_cast<void *>(cpu_input_buffers[i]),
            total_bytes);

      } else {
        int64_t total_size;
        total_size = output_bytes[i - input_names_.size()];
        VLOG(4) << "output_bytes: " << total_size;
        void *gpu_ptr;
        cudaMalloc(&gpu_ptr, total_size);
        engine_input_ptr[InputIndexToBindIndex_[i]] = gpu_ptr;
      }
    }

    clock_t startTime, endTime;
    startTime = clock();
    context->Execute(infer_batch, engine_input_ptr.data());
    endTime = clock();

    if (dlnne_log_flag_) {
      double during_ms =
          static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC * 1000;
      LOG(INFO) << "dlNNE execute time: " << during_ms << " ms";
    }

    bind_index = 0;
    for (unsigned int i = 0; i < engine_input_size; i++) {
      if (InputIndexToBindIndex_[i] < 0) continue;

      if (i >= input_names_.size()) {
        void *cpu_ptr = cpu_output_buffers[i - input_names_.size()];
        int64_t size;
        size = output_bytes[i - input_names_.size()];
        paddle::inference::CopyTensorDeviceToCpu(
            cpu_ptr, engine_input_ptr[InputIndexToBindIndex_[i]], size);

        cpu_output_buffers[bind_index] = cpu_ptr;

        if (dump_flag_) {
          std::stringstream dump_output_name;
          dump_output_name << engine_key_ << "_output_" << bind_index << ".txt";
          std::ofstream dump_output_file;
          dump_output_file.open(dump_output_name.str());
          for (int64_t i = 0; i < out_ele_nums[bind_index]; i++) {
            dump_output_file
                << static_cast<float *>(cpu_output_buffers[bind_index])[i]
                << "\n";
          }
          dump_output_file << "\b";
          dump_output_file.close();
        }
        bind_index++;
      }
      cudaFree(engine_input_ptr[InputIndexToBindIndex_[i]]);
    }
    dlnne_create_lock.unlock();
  }

  void RunNativeImpl(const framework::Scope &scope,
                     const platform::Place &dev_place) const {
    VLOG(4) << "RunNativeImpl";
    framework::Executor executor(dev_place);
    auto *block = Attr<framework::BlockDesc *>("sub_block");
    auto *program = block->Program();
    auto &current_scope = scope.NewScope();
    auto ctx = executor.Prepare(*program, block->ID());
    executor.RunPreparedContext(ctx.get(), &current_scope, false, true, true);
  }

  void RunCalibration(const framework::Scope &scope,
                      const platform::Place &dev_place) const {
    std::unordered_map<std::string, void *> calib_data_map;
    std::unordered_map<std::string, std::vector<int64_t>> calib_data_shape_map;
    std::unordered_map<std::string, std::string> calib_data_type_map;
    std::unordered_map<std::string, int64_t> calib_buffer_size_map;

    for (auto &x : Inputs("Xs")) {
      if (param_names_.count(x)) continue;
      auto &t =
          inference::analysis::GetFromScope<framework::LoDTensor>(scope, x);
      calib_data_map.emplace(x, t.data());

      // TODO(pei.jiang): refine this code, because when run dlnne create
      // engine, there is same code
      auto t_shape = phi::vectorize<int64_t>(t.dims());
      std::vector<int64_t> input_shape(t_shape.begin(), t_shape.end());
      calib_data_shape_map.emplace(x, input_shape);
      std::string data_type = inference::ConvertType(t.type());
      calib_data_type_map.emplace(x, data_type);

      int data_bytes = inference::GetDataByte(t.type());
      VLOG(4) << "input name: " << x << ", data_type: " << data_type;
      VLOG(4) << "data shape: ";
      int64_t buffer_size = data_bytes;
      for (auto dim : input_shape) {
        buffer_size *= dim;
        VLOG(4) << dim;
      }
      VLOG(4) << "buffer_size: " << buffer_size;
      calib_buffer_size_map.emplace(x, buffer_size);
    }

    std::string random_key = inference::GenerateRandomKey();
    for (auto calib_data : calib_data_map) {
      std::string input_name = calib_data.first;
      std::string input_data_path = calibration_data_path_ + "/" + input_name;
      MKDIR(input_data_path.c_str());

      std::string input_data_item_path =
          input_data_path + "/" + random_key + ".binary";
      auto outfile = std::fstream(input_data_item_path.c_str(),
                                  std::ios::out | std::ios::binary);
      int64_t buffer_size = calib_buffer_size_map[input_name];
      outfile.write(reinterpret_cast<char *>(calib_data.second), buffer_size);
      outfile.close();
    }

    std::stringstream calib_config_ss;
    calib_config_ss << "shape message: " << std::endl;
    for (auto const &shape_item : calib_data_shape_map) {
      calib_config_ss << shape_item.first << ":";
      for (auto const &dim : shape_item.second) {
        calib_config_ss << dim << " ";
      }
      calib_config_ss << std::endl;
    }

    calib_config_ss << "dtype message: " << std::endl;
    for (auto const &dtype_item : calib_data_type_map) {
      calib_config_ss << dtype_item.first << ":" << dtype_item.second
                      << std::endl;
    }

    std::ofstream calib_config_file;
    std::string calib_config_path =
        calibration_data_path_ + "/calib_config.txt";
    calib_config_file.open(calib_config_path);
    calib_config_file << calib_config_ss.str();
    calib_config_file.close();

    RunNativeImpl(scope, dev_place);
  }

  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    VLOG(4) << "calibration_mode_: " << calibration_mode_;
    if (calibration_mode_ == true) {
      VLOG(4) << "RunCalibration";
      RunCalibration(scope, dev_place);
      return;
    }

    RunDlnneOnCreateEngine(scope, dev_place);
  }
};

}  // namespace operators
}  // namespace paddle
