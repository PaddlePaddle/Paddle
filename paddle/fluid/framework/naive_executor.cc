// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/naive_executor.h"

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/denormal.h"
#ifdef PADDLE_WITH_DNNL
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#ifdef PADDLE_WITH_TENSORRT
#include "paddle/fluid/operators/tensorrt/tensorrt_engine_op.h"
#endif
#ifdef PADDLE_WITH_NVTX
#include "paddle/fluid/platform/device/gpu/cuda/cuda_profiler.h"
#endif
#ifdef PADDLE_WITH_LITE
#include "paddle/fluid/operators/lite/lite_engine_op.h"
#endif
#include "paddle/fluid/platform/flags.h"

PADDLE_DEFINE_EXPORTED_bool(
    naive_executor_sync_op,
    false,
    "Enable sync after each op run, used for debug.");
PADDLE_DEFINE_EXPORTED_bool(
    naive_executor_print_dims_after_op,
    false,
    "used for debug.");
PADDLE_DEFINE_EXPORTED_bool(
    naive_executor_print_value_after_op,
    false,
    "used for debug.");

namespace paddle {
namespace framework {
void NaiveExecutor::Prepare(Scope *scope,
                            const ProgramDesc &program_desc,
                            int block_id) {
  if (!scope) {
    scope_ = new framework::Scope;
  } else {
    scope_ = scope;
  }

  VLOG(3) << "NaiveExecutor init with scope " << scope;
  CreateOps(program_desc, block_id);
}

void NaiveExecutor::PrepareInterpreterCore(
    Scope *scope,
    const ProgramDesc &program_desc,
    const framework::interpreter::ExecutionConfig &execution_config) {
  interpreter_core_ = std::make_unique<framework::InterpreterCore>(
      place_, program_desc.Block(0), scope, execution_config);
}

void NaiveExecutor::PrepareInterpreterCore(
    Scope *scope,
    const ::pir::Program &pir_program,
    const framework::interpreter::ExecutionConfig &execution_config) {
  interpreter_core_ =
      std::make_unique<framework::InterpreterCore>(place_,
                                                   std::vector<std::string>{},
                                                   pir_program.block(),
                                                   scope,
                                                   execution_config);
}

void NaiveExecutor::RunInterpreterCore(
    const std::vector<std::string> &feed_names,
    bool need_fetch,
    bool switch_stream) {
  platform::ScopedFlushDenormal flush;
#ifdef PADDLE_WITH_NVTX
  platform::CudaNvtxRangePush("model", platform::NvtxRangeColor::Yellow);
#endif
  interpreter_core_->Run(feed_names, need_fetch, false, false, switch_stream);
#ifdef PADDLE_WITH_NVTX
  platform::CudaNvtxRangePop();
#endif
}

template <typename T>
void print(T *cpu_data, int num, std::string var_name) {
  std::string file_name = std::string("/root/paddlejob/workspace/env_run/zkk/cpp/") + var_name; 
  FILE* f = fopen(file_name.c_str(), "a+");
  for (int i = 0; i < num; i++) {
    fprintf(f,"%f\n", cpu_data[i]);
  }
  fclose(f);
}

void NaiveExecutor::Run() {
#ifdef PADDLE_WITH_DNNL
  platform::AttachPointerHashToMKLDNNKey(this, place_);
  platform::RegisterModelLayout(ops_, place_);
#endif
  platform::ScopedFlushDenormal flush;
#ifdef PADDLE_WITH_NVTX
  platform::CudaNvtxRangePush("model", platform::NvtxRangeColor::Yellow);
#endif
  for (auto &op : ops_) {
    VLOG(4) << std::this_thread::get_id() << " run "
            << op->DebugStringEx(scope_) << " on scope " << scope_;
    op->SetIsCalledByExecutor(false);


std::vector<std::string> result = {
  "data_preprocess",
  "img_bkb",
  "img_neck",
  "long_temp_fusion",
  "ldmap_routing_fusion",
  "ego_info_fusion",
  "cam2bev_modules",
  "2d_traffic_light",
  "pts_det_traj_head",
  "admap_head",
  "occ_head",
  "2d_occ_head",
  "pnc_head",
  "pnc_post_process",
  "traj_head",
};
bool nvtx = false;
std::string output_name = op->OutputVars(true).front();



for (auto ele:result) {
  if(output_name.find(ele) != std::string::npos) {
    nvtx = true;
  }
}
nvtx = true;
if(nvtx && 0){
  std::cout << 0 << std::endl;
}

    for (auto &func : input_hookfuncs_) {
      func(op.get(), scope_);
    }

    if (op->Type() == "while" || op->Type() == "conditional_block") {
      op->SetOutputHooks(output_hookfuncs_);
      op->SetInputHooks(input_hookfuncs_);
    }

#ifdef PADDLE_WITH_NVTX
    platform::CudaNvtxRangePush(op->Type() + "|" + op->OutputVars(true).front(),
                                platform::NvtxRangeColor::Green);
#endif

if (FLAGS_naive_executor_sync_op)
{
  std::cout << op->Type() << "run" << std::endl;
}

    // if (op->Type() == "while") {
    //   op->SetOutputHooks(hookfuncs_);
    // }

    op->Run(*scope_, place_);

  if (FLAGS_naive_executor_sync_op)
  {
    std::cout << op->OutputVars(true).front() << std::endl;
    paddle::platform::DeviceContextPool &pool = paddle::platform::DeviceContextPool::Instance();
    auto *dev_ctx = reinterpret_cast<phi::GPUContext *>(pool.Get(place_));
    auto success = cudaStreamSynchronize(dev_ctx->stream());
      std::cout <<  cudaGetErrorString( cudaGetLastError() ) << std::endl;
      PADDLE_ENFORCE_GPU_SUCCESS(success);


    if (FLAGS_naive_executor_print_dims_after_op) {
      for (auto name : op->OutputVars(true)) {
        std::cout << name << ": ";
        auto var = scope_->FindVar(name);
        if (!var->IsType<phi::DenseTensor>()) continue;
        auto tensor = FindTensor(name);
        if (tensor->numel() <= 0) continue;
        auto dims = tensor->dims();
        std::cout << "[";
        for (int64_t i = 0; i < dims.size(); i++) {
          std::cout << dims[i] << " ";
        }
        std::cout << "]";
        std::cout << std::endl;
        // 是否打印值
        if (FLAGS_naive_executor_print_value_after_op) {
          int want_num = std::min(100000, (int)(tensor->numel()));
          if (tensor->dtype() == paddle::DataType::FLOAT32) {
            float *cpu_data = new float[tensor->numel()];
            float *tensor_data = tensor->data<float>();
            if (tensor->place() == platform::CPUPlace()) {
              memcpy(cpu_data, tensor_data, sizeof(float) * want_num);
            } else {
              cudaMemcpy(cpu_data, tensor_data, sizeof(float) * want_num,
                        cudaMemcpyDeviceToHost);
            }
            float max_value = -10000;
            float min_value = 10000;

            for(int i = 0; i < want_num; i++) {
              if(cpu_data[i] > 10000 || cpu_data[i] < -10000) {
                if (cpu_data[i] > max_value) {
                  max_value = cpu_data[i];
                }
                if (cpu_data[i] < min_value) {
                  min_value = cpu_data[i];
                }
              }
            }
            if (max_value > 10000) {
              std::cout << "输出最大值：" << max_value << std::endl;
            }
            if (min_value < -10000) {
              std::cout << "输出最小值：" << min_value << std::endl;
            }
            delete[] cpu_data;
          }
          
          want_num = 1;
          if (tensor->dtype() == paddle::DataType::INT32) {
            int *cpu_data = new int[want_num];
            int *tensor_data = tensor->data<int>();
            if (tensor->place() == platform::CPUPlace()) {
              memcpy(cpu_data, tensor_data, sizeof(int) * want_num);
            } else {
              cudaMemcpy(cpu_data, tensor_data, sizeof(int) * want_num,
                        cudaMemcpyDeviceToHost);
            }
            
            for(int i = 0; i < want_num; i++) {
              std::cout << "int32 数字" << cpu_data[i] << std::endl;
            }
            delete[] cpu_data;
          }
          if (tensor->dtype() == paddle::DataType::INT64) {
            int64_t *cpu_data = new int64_t[want_num];
            int64_t *tensor_data = tensor->data<int64_t>();
            if (tensor->place() == platform::CPUPlace()) {
              memcpy(cpu_data, tensor_data, sizeof(int64_t) * want_num);
            } else {
              cudaMemcpy(cpu_data, tensor_data, sizeof(int64_t) * want_num,
                        cudaMemcpyDeviceToHost);
            }
            
            for(int i = 0; i < want_num; i++) {
              std::cout << "int64 数字" << cpu_data[i] << std::endl;
            }
            delete[] cpu_data;
          }
        }
      }
    }
  }

#ifdef PADDLE_WITH_NVTX
    platform::CudaNvtxRangePop();
#endif

    // Update the shared_holder so that only records the max one.
    if (reuse_cache_.count(op.get())) {
      for (auto &it : reuse_cache_[op.get()]) {
        if (it.first->memory_size() >
            cluster_buffer_[it.second]->memory_size()) {
          cluster_buffer_[it.second] = it.first;
          int updated_cluster_id = it.second;

          // cluster_buffer_[it.second] has been updated to be a new
          // phi::DenseTensor*, we need change all phi::DenseTensor's
          // shared_holder in this cluster. The following two loops code looks
          // ugly, it does work. The following two loops seem time-consuming,
          // but once the memory reaches its peak, the cluster will not update,
          // so it's ok.
          for (auto &op_map : reuse_cache_) {
            // op_map.second is std::unordered_map<phi::DenseTensor*, int>.
            for (auto &it2 : op_map.second) {
              if (it2.second == updated_cluster_id) {
                it2.first->ShareBufferWith(*cluster_buffer_[it2.second], true);
              }
            }
          }
        }
      }
    }

    for (auto &func : output_hookfuncs_) {
      func(op.get(), scope_);
    }
  }
#ifdef PADDLE_WITH_NVTX
  platform::CudaNvtxRangePop();
#endif
}

void NaiveExecutor::CreateVariables(const ProgramDesc &desc,
                                    int block_id,
                                    bool persistable,
                                    Scope *scope) {
  PADDLE_ENFORCE_NOT_NULL(scope,
                          platform::errors::InvalidArgument(
                              "The Scope to hold variables is nullptr."));

  auto &global_block = desc.Block(block_id);

  const auto *anc = scope;
  PADDLE_ENFORCE_NE(
      anc->parent(),
      anc,
      platform::errors::InvalidArgument("Input scope should be child scope."));
  while (anc->parent()) {
    anc = anc->parent();
  }

  int num_vars = 0;
  for (auto &var : global_block.AllVars()) {
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }
    num_vars++;

    if (persistable == var->Persistable()) {
      if (persistable) {
        if (!anc->FindVar(var->Name())) {
          auto *ptr = const_cast<Scope *>(anc)->Var(var->Name());
          VLOG(3) << scope << " Create persistable variable " << var->Name()
                  << ", which pointer is " << ptr;
          InitializeVariable(ptr, var->GetType());
        }
      } else {
        auto *ptr = const_cast<Scope *>(scope)->Var(var->Name());
        VLOG(3) << scope << " Create variable " << var->Name()
                << ", which pointer is " << ptr;
        InitializeVariable(ptr, var->GetType());
      }
    }
  }
  VLOG(4) << "naive executor create " << num_vars << " vars";
}

void NaiveExecutor::CreateOps(const ProgramDesc &desc, int block_id) {
  for (const auto &op_desc : desc.Block(block_id).AllOps()) {
    if (op_desc->Type() == "feed" || op_desc->Type() == "fetch") {
      LOG(INFO) << "---  skip [" << op_desc->Input("X")[0] << "], "
                << op_desc->Type() << " -> " << op_desc->Output("Out")[0];
      continue;
    }
    ops_.emplace_back(OpRegistry::CreateOp(*op_desc));
  }
}

phi::DenseTensor *NaiveExecutor::FindTensor(const std::string &name) {
  PADDLE_ENFORCE_NOT_NULL(scope_,
                          platform::errors::PreconditionNotMet(
                              "Need to init scope in NaiveExecutor firstly."));
  auto *var = scope_->FindVar(name);
  PADDLE_ENFORCE_NOT_NULL(
      var,
      platform::errors::NotFound("No variable [%s] in current scope.", name));
  auto *tensor = const_cast<phi::DenseTensor *>(&var->Get<phi::DenseTensor>());
  return tensor;
}

void NaiveExecutor::RegisterOutputHook(const HookFunc &hookfunc) {
  output_hookfuncs_.push_back(hookfunc);
  if (interpreter_core_) {
    interpreter_core_->SetOutputHooks(output_hookfuncs_);
  }
}

void NaiveExecutor::RegisterInputHook(const HookFunc &hookfunc) {
  input_hookfuncs_.push_back(hookfunc);
  if (interpreter_core_) {
    interpreter_core_->SetInputHooks(input_hookfuncs_);
  }
}

void NaiveExecutor::MakeReusePlan(
    const std::unordered_map<std::string, std::string> &reuse_table) {
  std::unordered_map<std::string, std::unordered_set<std::string>> clusters;
  for (auto &it : reuse_table) {
    clusters[it.second].insert(it.first);
  }

  std::vector<std::string> cluster_names;
  for (auto &it : clusters) {
    cluster_names.push_back(it.first);
  }
  cluster_buffer_.resize(cluster_names.size());

  for (auto &op : ops_) {
    for (auto &name : op->OutputVars(true)) {
      if (reuse_table.count(name)) {
        const auto &reuse_name = reuse_table.at(name);
        auto it =
            std::find(cluster_names.begin(), cluster_names.end(), reuse_name);
        int idx = static_cast<int>(it - cluster_names.begin());
        auto *var = scope_->FindVar(name);
        auto *reuse_var = scope_->FindVar(reuse_name);
        if (var && reuse_var && var->IsType<phi::DenseTensor>() &&
            reuse_var->IsType<phi::DenseTensor>()) {
          auto *tensor = var->GetMutable<phi::DenseTensor>();
          auto *reuse_tensor = reuse_var->GetMutable<phi::DenseTensor>();
          cluster_buffer_[idx] = reuse_tensor;
          if (reuse_cache_.count(op.get())) {
            reuse_cache_[op.get()].emplace(tensor, idx);
          } else {
            reuse_cache_[op.get()] =
                std::unordered_map<phi::DenseTensor *, int>{{tensor, idx}};
          }
        }
      }
    }
  }
}

NaiveExecutor::~NaiveExecutor() {
#ifdef PADDLE_WITH_DNNL
  // Clear mkl-dnn cache,
  // this is needed to have mkl-dnn unit tests working
  platform::ClearMKLDNNCache(place_, this);
#endif
}

void NaiveExecutor::ResetTrtOps(int num) {
#ifdef PADDLE_WITH_TENSORRT
  for (auto &op : ops_) {
    if (op->Type() == "tensorrt_engine") {
      operators::TensorRTEngineOp *trtop =
          dynamic_cast<operators::TensorRTEngineOp *>(op.get());
      if (!trtop) return;
      std::string engine_key = trtop->Attr<std::string>("engine_key");
      int engine_predictor_id = trtop->Attr<int>("predictor_id");
      std::string engine_name =
          engine_key + std::to_string(engine_predictor_id);
      operators::TensorRTEngine *trt_engine = nullptr;
      // can't get trt engine if int8 calibration table data process.
      if (paddle::inference::Singleton<
              inference::tensorrt::TRTEngineManager>::Global()
              .Has(engine_name)) {
        trt_engine = paddle::inference::Singleton<
                         inference::tensorrt::TRTEngineManager>::Global()
                         .Get(engine_name);
      }
      if (trt_engine && trt_engine->with_dynamic_shape()) {
        LOG(INFO) << "rebuild trt engine, this may cost a lot of time!";
        trt_engine->ResetContext();
        trt_engine->ClearTensorMap();
        trt_engine->SetProfileNum(num);
        auto *anc = scope_->parent();
        while (anc && anc->parent()) {
          anc = anc->parent();
        }
        if (anc == nullptr) {
          anc = scope_;
        }
        trtop->PrepareTRTEngine(*anc, trt_engine);
      }
    }
  }
#endif
}

void NaiveExecutor::CloneLiteEngine(int num, void *stream) {
#ifdef PADDLE_WITH_LITE
  for (auto &op : ops_) {
    if (op->Type() == "lite_engine") {
      operators::LiteEngineOp *lite_op =
          dynamic_cast<operators::LiteEngineOp *>(op.get());
      PADDLE_ENFORCE_NOT_NULL(
          lite_op,
          phi::errors::InvalidArgument(
              "lite_op(type: lite_engine) should be created."));
      std::string engine_key = lite_op->Attr<std::string>("engine_key");
      std::string new_engine_key = engine_key + "_" + std::to_string(num);
      PADDLE_ENFORCE(
          paddle::inference::Singleton<inference::lite::EngineManager>::Global()
              .Has(engine_key),
          phi::errors::InvalidArgument(
              "lite_engine(key: %s) should be created.", engine_key));
      auto *lite_engine =
          paddle::inference::Singleton<inference::lite::EngineManager>::Global()
              .Get(engine_key);
      auto new_lite_engine = lite_engine->Clone();
#ifdef LITE_SUBGRAPH_WITH_XPU
      new_lite_engine->SetStream(TARGET(kXPU), stream);
#endif
      paddle::inference::Singleton<inference::lite::EngineManager>::Global()
          .Set(new_engine_key, new_lite_engine);
      lite_op->SetAttr("engine_key", new_engine_key);
      lite_op->SetEngine(new_lite_engine.get());
    }
  }
#endif
}

}  // namespace framework
}  // namespace paddle
