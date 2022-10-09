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

#include "paddle/fluid/framework/ir/pass_builder.h"

#include <memory>

#include "glog/logging.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

class Pass;

using VarQuantScale =
    std::unordered_map<std::string, std::pair<bool, LoDTensor>>;

std::shared_ptr<Pass> PassBuilder::AppendPass(const std::string& pass_name) {
  VLOG(1) << "Append " << pass_name;
  auto pass = ir::PassRegistry::Instance().Get(pass_name);

  // pass->Set("use_varseqlen", false));
  // pass->Set("with_interleaved",
  //           false);
  // pass->Set("tensorrt_transformer_posid",
  //           new std::string(argument->tensorrt_transformer_posid()));
  // pass->Set("tensorrt_transformer_maskid",
  //           new std::string(argument->tensorrt_transformer_maskid()));
  // pass->Set("disable_logs", new bool(argument->disable_logs()));
  // auto precision_mode = argument->tensorrt_precision_mode();
  // bool enable_int8 = precision_mode == AnalysisConfig::Precision::kInt8;
  // pass->Set("enable_int8", new bool(enable_int8));
  // pass->Set("max_input_shape",
  //           new std::map<std::string, std::vector<int>>(
  //               argument->max_input_shape()));
  // pass->Set("min_input_shape",
  //           new std::map<std::string, std::vector<int>>(
  //               argument->min_input_shape()));
  // pass->Set("optim_input_shape",
  //           new std::map<std::string, std::vector<int>>(
  //               argument->optim_input_shape()));
  // // tuned trt dynamic_shape
  // pass->Set("trt_tuned_dynamic_shape",
  //           new bool(argument->tensorrt_tuned_dynamic_shape()));
  // bool with_dynamic_shape = (argument->max_input_shape().size() > 0 &&
  //                            argument->min_input_shape().size() > 0 &&
  //                            argument->optim_input_shape().size() > 0) ||
  //                           argument->tensorrt_tuned_dynamic_shape();
  // pass->Set("with_dynamic_shape", new bool(with_dynamic_shape));

  // pass->Set("model_precision", new int(argument->model_precision()));
  // pass->Set(
  //     "mixed_black_list",
  //     new std::unordered_set<std::string>(argument->mixed_black_list()));

#ifdef PADDLE_WITH_MKLDNN
  if (pass_name == "cpu_quantize_pass") {
    // if (argument->quantize_enabled_op_types().count("conv2d") ||
    //     argument->quantize_enabled_op_types().count("depthwise_conv2d")) {
    //   pass->Set("data_layout", new std::string("NHWC"));
    // }
    pass->Set("quant_var_scales", new VarQuantScale({}));
  }
#endif

  if (pass_name == "fc_fuse_pass") {
    pass->Set("use_gpu", new bool(false));
    pass->Set("use_fc_padding", new bool(false));
  }

  passes_.emplace_back(std::move(pass));
  return passes_.back();
}

void PassBuilder::RemovePass(size_t idx) {
  PADDLE_ENFORCE_GT(
      passes_.size(),
      idx,
      platform::errors::InvalidArgument(
          "Passes size is %d, %d is not a valid index.", passes_.size(), idx));
  passes_.erase(passes_.begin() + idx);
}

std::shared_ptr<Pass> PassBuilder::InsertPass(size_t idx,
                                              const std::string& pass_type) {
  PADDLE_ENFORCE_GE(
      passes_.size(),
      idx,
      platform::errors::InvalidArgument(
          "Passes size is %d, %d is not a valid index.", passes_.size(), idx));
  std::shared_ptr<Pass> pass(
      ir::PassRegistry::Instance().Get(pass_type).release());
  passes_.insert(passes_.begin() + idx, std::move(pass));
  return passes_[idx];
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle
