/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/distributed/parameter_prefetch.h"
#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

template <typename DeviceContext, typename T>
class DistributedLookupTableKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto ids_vars = context.MultiInputVar("Ids");
    auto emb_vars = context.MultiOutput<framework::Tensor>("Embeddings");

    auto id_names = context.InputNames("Ids");
    auto embedding_name = context.InputNames("W").front();
    auto out_names = context.OutputNames("Outputs");
    auto lookup_tables = context.Attr<std::vector<std::string>>("table_names");
    auto endpoints = context.Attr<std::vector<std::string>>("endpoints");
    auto is_distributed = context.Attr<bool>("is_distributed");

    auto lookup_table_version =
        context.Attr<std::string>("lookup_table_version");

    operators::distributed::prefetchs(id_names, out_names, embedding_name,
                                      is_distributed, lookup_tables, endpoints,
                                      context, context.scope());

    if (lookup_table_version == "lookup_table_v2") {
      auto &scope = context.scope();
      auto emb_dim =
          scope.FindVar(embedding_name)->Get<framework::LoDTensor>().dims()[1];

      for (size_t i = 0; i < id_names.size(); ++i) {
        auto *id_var = scope.FindVar(id_names[i]);
        auto *out_var = scope.FindVar(out_names[i]);
        auto *id_tensor = id_var->GetMutable<framework::LoDTensor>();
        auto *out_tensor = out_var->GetMutable<framework::LoDTensor>();

        auto id_dims = id_tensor->dims();
        out_tensor->Resize(framework::make_ddim(
            {static_cast<int64_t>(id_dims[0]), static_cast<int64_t>(id_dims[1]),
             static_cast<int64_t>(emb_dim)}));
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
