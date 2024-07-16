/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/new_executor/collect_shape_manager.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/kernels/funcs/data_type_transform.h"

namespace paddle {
namespace framework {
CollectShapeManager &CollectShapeManager::Instance() {
  static CollectShapeManager instance;
  return instance;
}

void CollectShapeManager::CollectShapeInfo(
    framework::InstructionBase *instr,
    framework::ValueExecutionInfo *value_exe_info,
    framework::Scope *scope) {
  is_shape_range_info_ready_ = false;
  for (auto &input : instr->Inputs()) {
    auto var_name = value_exe_info->GetVarName(input.first);
    auto *var = scope->FindVar(var_name);
    if (!var || !var->IsType<phi::DenseTensor>()) continue;
    auto tensor = var->Get<phi::DenseTensor>();
    if (!tensor.initialized()) continue;
    paddle::platform::DeviceContextPool &pool =
        paddle::platform::DeviceContextPool::Instance();
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto *dev_ctx = pool.Get(phi::GPUPlace());
    auto stream = static_cast<phi::GPUContext *>(dev_ctx)->stream();
#ifdef PADDLE_WITH_HIP
    hipStreamSynchronize(stream);
#else
    cudaStreamSynchronize(stream);
#endif
#endif

    framework::DDim dim = tensor.dims();
    std::vector<int32_t> shape(dim.size());
    for (int i = 0; i < static_cast<int>(shape.size()); ++i)
      shape[i] = static_cast<int32_t>(dim[i]);
    if (!shape.empty()) {
      shape_info_[input.first].emplace_back(shape);
    } else if (tensor.numel() > 0) {
      // This must be a zero dimension tensor.
      PADDLE_ENFORCE_EQ(tensor.numel(),
                        1UL,
                        platform::errors::PreconditionNotMet(
                            "This tensor must have one element, but got %ld.",
                            tensor.numel()));
      std::vector<int32_t> zero_shape(1, 1);
      shape_info_[input.first].emplace_back(zero_shape);
    }

    // We need collect value range for shape tensor for Paddle-TRT's use.
    // To be noticed, this method to identify all shape tensors is based on
    // assumption that all shape tensors in the model have numbers <= 8.
    // This is a simple method to identify all shape tensors with some
    // mistakes, but it doesn't matter.
    auto is_shape_tensor = tensor.numel() <= 8 && tensor.numel() >= 1;
    if ((tensor.dtype() == phi::DataType::INT32 ||
         tensor.dtype() == phi::DataType::INT64) &&
        is_shape_tensor) {
      std::vector<int> int32_host(tensor.numel());

      if (platform::is_cpu_place(tensor.place())) {
        auto &int32_tensor = tensor;
        if (tensor.dtype() == phi::DataType::INT64) {
          auto *cpu_ctx = pool.Get(platform::CPUPlace());
          int32_tensor = phi::funcs::TransDataType(
              reinterpret_cast<const phi::CPUContext &>(*cpu_ctx),
              tensor,
              DataType::INT32);
        }
        paddle::memory::Copy(platform::CPUPlace(),
                             int32_host.data(),
                             platform::CPUPlace(),
                             int32_tensor.data<int>(),
                             int32_tensor.numel() * sizeof(int));
      } else if (platform::is_gpu_place(tensor.place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
        auto *dev_ctx = pool.Get(tensor.place());
        auto &int32_tensor = tensor;
        if (tensor.dtype() == phi::DataType::INT64) {
          int32_tensor = phi::funcs::TransDataType(
              reinterpret_cast<const phi::GPUContext &>(*dev_ctx),
              tensor,
              DataType::INT32);
        }
        paddle::memory::Copy(platform::CPUPlace(),
                             int32_host.data(),
                             int32_tensor.place(),
                             int32_tensor.data<int>(),
                             int32_tensor.numel() * sizeof(int),
                             nullptr);
#endif
      }
      shape_tensor_info_[input.first].emplace_back(int32_host);
    }
  }
}

void CollectShapeManager::StatisticShapeRangeInfo() {
  if (is_shape_range_info_ready_) {
    return;
  }
  auto extract_min_max_opt =
      [](std::map<pir::Value, std::vector<int32_t>> &min_data,
         decltype(min_data) max_data,
         decltype(min_data) opt_data,
         decltype(shape_info_) shape_data) {
        for (auto const &it : shape_data) {
          auto val = it.first;
          auto shapes = it.second;
          std::vector<int32_t> min_shape(shapes[0].begin(), shapes[0].end());
          std::vector<int32_t> max_shape(shapes[0].begin(), shapes[0].end());
          std::vector<int32_t> opt_shape(shapes[0].begin(), shapes[0].end());

          auto ShapeMaxFreq =
              [](const std::map<int32_t, int32_t> &m) -> int32_t {
            std::vector<std::pair<int32_t, int32_t>> counter;
            for (auto &it : m) counter.emplace_back(it);
            std::sort(counter.begin(),
                      counter.end(),
                      [](std::pair<int32_t, int32_t> &a,
                         std::pair<int32_t, int32_t> &b) {
                        return a.second > b.second;
                      });
            return counter[0].first;
          };

          for (size_t d = 0; d < shapes[0].size(); ++d) {
            std::map<int32_t, int32_t> counter;
            for (auto &shape : shapes) {
              counter[shape[d]] += 1;
              if (shape[d] < min_shape[d]) min_shape[d] = shape[d];
              if (shape[d] > max_shape[d]) max_shape[d] = shape[d];
            }
            opt_shape[d] = ShapeMaxFreq(counter);
          }

          min_data[val] = min_shape;
          max_data[val] = max_shape;
          opt_data[val] = opt_shape;
        }
      };
  extract_min_max_opt(min_shapes_, max_shapes_, opt_shapes_, shape_info_);
  extract_min_max_opt(
      min_values_, max_values_, opt_values_, shape_tensor_info_);
  is_shape_range_info_ready_ = true;
}

std::vector<int32_t> CollectShapeManager::GetValueShapeRangeInfo(
    pir::Value op_val, bool is_shape_tensor, ShapeMode shape_mode) {
  PADDLE_ENFORCE_EQ(is_shape_range_info_ready_,
                    true,
                    ::common::errors::PreconditionNotMet(
                        "Shape range info has not been calculated and "
                        "StatisticShapeRangeInfo must be called first."));
  PADDLE_ENFORCE_NE(op_value2kernel_value_.find(op_val),
                    op_value2kernel_value_.end(),
                    ::common::errors::NotFound(
                        "Can't find kernel_value that corresponding to "
                        "op_value, maybe origin program has changed or not "
                        "open FLAGS_enable_collect_shape."));
  auto kernel_val = op_value2kernel_value_[op_val];
  if (shape_mode == ShapeMode::kMIN) {
    if (is_shape_tensor) {
      PADDLE_ENFORCE_NE(
          min_values_.find(kernel_val),
          min_values_.end(),
          ::common::errors::NotFound("Can't find min shape according to the "
                                     "input Value that is a shape tensor."));
      return min_values_[kernel_val];
    } else {
      PADDLE_ENFORCE_NE(
          min_shapes_.find(kernel_val),
          min_shapes_.end(),
          ::common::errors::NotFound("Can't find min shape according to the "
                                     "input Value that isn't a shape tensor"));
      return min_shapes_[kernel_val];
    }
  } else if (shape_mode == ShapeMode::kMAX) {
    if (is_shape_tensor) {
      PADDLE_ENFORCE_NE(
          max_values_.find(kernel_val),
          max_values_.end(),
          ::common::errors::NotFound("Can't find max shape according to the "
                                     "input Value that is a shape tensor."));
      return max_values_[kernel_val];
    } else {
      PADDLE_ENFORCE_NE(
          max_shapes_.find(kernel_val),
          max_shapes_.end(),
          ::common::errors::NotFound("Can't find max shape according to the "
                                     "input Value that isn't a shape tensor"));
      return max_shapes_[kernel_val];
    }
  } else if (shape_mode == ShapeMode::kOPT) {
    if (is_shape_tensor) {
      PADDLE_ENFORCE_NE(
          opt_values_.find(kernel_val),
          opt_values_.end(),
          ::common::errors::NotFound("Can't find opt shape according to the "
                                     "input Value that is a shape tensor."));
      return opt_values_[kernel_val];
    } else {
      PADDLE_ENFORCE_NE(
          opt_shapes_.find(kernel_val),
          opt_shapes_.end(),
          ::common::errors::NotFound("Can't find opt shape according to the "
                                     "input Value that isn't a shape tensor"));
      return opt_shapes_[kernel_val];
    }
  } else {
    PADDLE_THROW(phi::errors::Unimplemented(
        "We only support ShapeMode::kMIN, ShapeMode::kMax and ShapeMode::kOpt "
        "when GetValueShapeRangeInfo"));
  }
}

}  // namespace framework
}  // namespace paddle
