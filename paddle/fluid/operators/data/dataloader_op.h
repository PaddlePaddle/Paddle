/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/data/pipeline.h"

namespace paddle {
namespace operators {

class Pipeline;

template <typename DeviceContext, typename T>
class DataLoaderOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    LOG(ERROR) << "DataLoaderOpKernel enter";
    // Step1: get output vars and attrs
    auto output_vars = ctx.MultiOutputVar("Out");
    auto output_var_names = ctx.OutputNames("Out");

    auto* global_block = ctx.Attr<BlockDesc*>("global_block");
    auto start_op_index = ctx.Attr<int64_t>("start_op_index");
    auto end_op_index = ctx.Attr<int64_t>("end_op_index");
    auto program_id = ctx.Attr<int64_t>("program_id");

    auto pipeline = data::PipelineManager::Instance()->GetPipeline(
        program_id, global_block, ctx.GetPlace(), start_op_index, end_op_index,
        output_var_names);

    pipeline->ReadNext(output_vars);

    if (!pipeline->IsRunning()) {
      LOG(ERROR) << "DataLoaderOpKernel Pipeline not running";
      data::PipelineManager::Instance()->ShutDownPipeline(program_id);
      throw platform::EOFException("DataLoaderOpKernel epoch end",
                                    __FILE__, __LINE__);
    }

    LOG(ERROR) << "DataLoaderOpKernel finish";
  }
};

}  // namespace operators
}  // namespace paddle
