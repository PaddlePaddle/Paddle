// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/ir/group_schedule/search/measurer.h"
#include <sstream>

#include "glog/logging.h"
#include "paddle/cinn/hlir/dialect/operator/ir/op_dialect.h"
#include "paddle/cinn/hlir/dialect/operator/transforms/add_cinn_pass.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/transforms/build_cinn_pass.h"
#include "paddle/fluid/pir/transforms/pd_op_to_kernel_pass.h"
#include "paddle/phi/backends/context_pool.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/pir/include/core/builtin_dialect.h"
#include "paddle/pir/include/core/builtin_type.h"
#include "paddle/pir/include/core/program.h"
#include "paddle/pir/include/dialect/shape/ir/shape_dialect.h"
#include "paddle/pir/include/pass/pass.h"
#include "paddle/pir/include/pass/pass_manager.h"

COMMON_DECLARE_bool(print_ir);
PD_DECLARE_string(cinn_kernel_execution_label);

namespace cinn {
namespace ir {
namespace search {

std::shared_ptr<pir::PassManager> CreatePassManager() {
  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<cinn::dialect::OperatorDialect>();
  ctx->GetOrRegisterDialect<pir::shape::ShapeDialect>();
  auto pass_manager = std::make_shared<pir::PassManager>(ctx);
  if (FLAGS_print_ir) {
    pass_manager->EnableIRPrinting();
  }
  return pass_manager;
}

Measurer::Measurer(::pir::Program* program) : main_program_(program) {
  std::stringstream ss;
  ss << *main_program_;
  compile_label_ = "Compile Main Program\n" + ss.str();
  execute_label_ = "Execute Main Program\n" + ss.str();
}

Measurer::Measurer(
  ::pir::Program* main_program,
  ::pir::Program* startup_program
) : main_program_(main_program), startup_program_(startup_program) {
  
  std::stringstream ss;
  ss << *main_program;
  compile_label_ = "Compile Main Program\n" + ss.str();
  execute_label_ = "Execute Main Program\n" + ss.str();
}

void Measurer::Prepare() {
  VLOG(4) << "[Debug] ============= Measurer::Prepare() Begin ============= \n";
  if (startup_program_ == nullptr) {
  VLOG(4) << "[Debug] Measurer::Prepare() Skip due to startup_program_ == nullptr";
    return;
  }
  ::pir::IrMapping ir_mapping;
  VLOG(4) << "[Debug] Measurer::Prepare() Before Clone ir_mapping";
  std::shared_ptr<::pir::Program> program_cloned = startup_program_->Clone(ir_mapping);
  VLOG(4) << "[Debug] Measurer::Prepare() Before ApplyCinnPass";
  cinn::dialect::ir::ApplyCinnPass(program_cloned.get(), CreatePassManager);
  VLOG(4) << "[Debug] Measurer::Prepare() Before PdOpLowerToKernelPass";
  kernel_program_ = std::move(
      paddle::dialect::PdOpLowerToKernelPass(program_cloned.get(), place_));
  VLOG(4) << "[Debug] Measurer::Prepare() Before executor_.reset";
  std::vector<std::string> fetch_var_names{};
  executor_.reset(new paddle::framework::InterpreterCore(
      place_, fetch_var_names, kernel_program_->block(), exe_scope_.get()));
  VLOG(4) << "[Debug] Measurer::Prepare() Before executor_.Run";
  std::vector<std::string> feed_names{};
  executor_->Run(feed_names, true);
  VLOG(4) << "[Debug] ============= Measurer::Prepare() End ============= \n";

}

void Measurer::Compile() {
  Prepare();
  // auto w0 = exe_scope_->FindVar("conv2d_0.w_0")->Get<phi::DenseTensor>();
  // const float* w0_cpu = w0.data<float>();
  // std::stringstream ss;
  // ss << "w0 = [";
  // for (size_t i = 0; i < w0.numel(); ++i) {
  //   ss << w0_cpu[i] << ", ";
  // }
  // ss << " ]";
  // VLOG(6) << ss.str();


  common::PerformanceStatisticsStart(compile_label_);
  ::pir::IrMapping ir_mapping;
  VLOG(4) << "[Debug] Measurer::Compile() Before Clone ir_mapping";
  std::shared_ptr<::pir::Program> program_cloned = main_program_->Clone(ir_mapping);
  VLOG(4) << "[Debug] Measurer::Compile() Before ApplyCinnPass";
  cinn::dialect::ir::ApplyCinnPass(program_cloned.get(), CreatePassManager);
  VLOG(4) << "[Debug] Measurer::Compile() Before PdOpLowerToKernelPass";
  kernel_program_ = std::move(
      paddle::dialect::PdOpLowerToKernelPass(program_cloned.get(), place_));
  VLOG(4) << "[Debug] Measurer::Compile() Before executor_.reset";
  executor_.reset(new paddle::framework::InterpreterCore(
      place_, {"out@fetch"}, kernel_program_->block(), exe_scope_.get()));
  VLOG(4) << "[Debug] Measurer::Compile() Kernel after Measurer compile: \n"<< *kernel_program_;
  common::PerformanceStatisticsEnd(compile_label_);
}

std::string ConcatShapeAsLabel(
    const std::unordered_map<std::string, std::vector<int64_t>>&
        input_name_and_shape) {
  std::stringstream ss;
  ss << "Shape  ";
  for (const auto item : input_name_and_shape) {
    ss << item.first << "=";
    for (int n : item.second) {
      ss << n << "x";
    }
  }
  std::string label = ss.str();
  label.pop_back();
  return label;
}

template <typename T> 
void CopyBetweenDeviceHost(phi::DenseTensor* dst, const phi::DenseTensor* src) {
  #if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  auto src_place = src->place();

  // auto *dev_ctxs = reinterpret_cast<const std::map<
  //     phi::Place,
  //     std::shared_future<std::unique_ptr<phi::DeviceContext>>> *>(
  //     device_contexts_);
  // auto *dev_ctx =
  //     static_cast<phi::GPUContext *>(dev_ctxs->at(src_place).get().get());
  auto * dev_ctx = static_cast<phi::GPUContext *>(phi::DeviceContextPool::Instance().Get(
    phi::GPUPlace()));
  paddle::memory::Copy(dst->place(),
                       static_cast<void *>(dst->data<T>()),
                       src_place,
                       src->data<T>(),
                       src->numel() * sizeof(T),
                       dev_ctx->stream());
#ifdef PADDLE_WITH_HIP
  hipStreamSynchronize(dev_ctx->stream());
#else

  cudaStreamSynchronize(dev_ctx->stream());
  
#endif
#else
  PADDLE_THROW(common::errors::Unavailable(
      "Can not create tensor with CUDA place because paddle is not compiled "
      "with CUDA."));
#endif
};


void Measurer::Run(const std::unordered_map<std::string, std::vector<int64_t>>&
                       input_name_and_shape,
                   int repeat) {
  std::vector<std::string> input_names;
  std::vector<phi::DenseTensor> input_tensors;
  for (const auto item : input_name_and_shape) {
    // LOG(INFO) << "input_name: " << item.first;

    // for (int i = 0; i < item.second.size(); ++i) {
    //   LOG(INFO) << "dim[" << i << "]: " << item.second[i]; 
    // }
    input_names.push_back(item.first);
    auto tensor =
        executor_->local_scope()->FindVar(item.first)->Get<phi::DenseTensor>();
    phi::DDim ddim(item.second.data(), item.second.size());
    tensor.ResizeAndAllocate(ddim);
    float* data = tensor.mutable_data<float>(ddim, place_);


    phi::DenseTensor cpu_tensor;
    cpu_tensor.ResizeAndAllocate(ddim);
    auto cpu = cpu_tensor.mutable_data<float>(ddim, phi::CPUPlace());

    for (size_t i = 0; i < cpu_tensor.numel(); ++i) {
      cpu[i] = static_cast<float>(sin(i));
    }

    CopyBetweenDeviceHost<float>(&tensor, &cpu_tensor);

    VLOG(3) << "[Debug] tensor.strides: " <<  cpu_tensor.strides();
    VLOG(3) << "[Debug] tensor.dims: " <<  cpu_tensor.dims();
    VLOG(3) << "[Debug] tensor.numel: " <<  cpu_tensor.numel();
    // VLOG(3) << "tensor.NumElements: " <<  tensor.NumElements();

    VLOG(3) << "[Debug] Input Tensor: " <<  paddle::framework::PrintDenseTensor(&cpu_tensor,0, 50);

    // LOG(INFO) << "data[0][0][0][0]" << data[0]; 
    input_tensors.push_back(tensor);
  }
  std::string input_shape_label = ConcatShapeAsLabel(input_name_and_shape);

  common::PerformanceStatistician& ps =
      common::PerformanceStatistician::Instance();
  for (int i = 0; i < repeat; ++i) {
    ps.Start(execute_label_ + "\n" + input_shape_label);
    auto fetch_list = executor_->Run(input_names, input_tensors, true);
    ps.End(execute_label_ + "\n" + input_shape_label);
    // for (size_t i = 0; i < fetch_list.size(); ++i) {
    //   const float* fetch_data =
    //   PADDLE_GET_CONST(phi::DenseTensor, fetch_list[i]).data<float>();
    //   VLOG(7) << "fetch_data[" << i << "] =  " << *fetch_data;
    // }
    auto tensor = PADDLE_GET_CONST(phi::DenseTensor, fetch_list[0]);
    VLOG(3) << "[Debug] Output Tensor: " <<  paddle::framework::PrintDenseTensor(&tensor,0, 50);

  }
}

MeasureResult Measurer::Result() const {
  MeasureResult result;
  common::PerformanceStatistician& ps =
      common::PerformanceStatistician::Instance();

  auto compile_durations =
      ::common::PerformanceReporter::ExtractDuration(ps.Record(compile_label_));
  auto total_execute_durations = ::common::PerformanceReporter::ExtractDuration(
      ps.RecordWithSubLabel(execute_label_));
  auto kernel_record = ps.Record(FLAGS_cinn_kernel_execution_label);
  auto kernel_execute_durations =
      ::common::PerformanceReporter::ExtractDuration(kernel_record);

  auto compile_time = ::common::PerformanceReporter::Mean(compile_durations);
  auto avg_total_execute_time =
      ::common::PerformanceReporter::Mean(total_execute_durations);
  VLOG(6) << " report: "
          << ::common::PerformanceReporter::Report(kernel_record);
  auto avg_kernel_execute_time =
      ::common::PerformanceReporter::TrimMean(kernel_execute_durations);

  result.compile_time = compile_time;
  result.avg_total_execute_time = avg_total_execute_time;
  result.avg_kernel_execute_time = avg_kernel_execute_time;

  ps.Reset();
  return result;
}

}  // namespace search
}  // namespace ir
}  // namespace cinn
