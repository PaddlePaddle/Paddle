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

#include "paddle/fluid/framework/details/graph_builder_factory.h"
#include <fstream>
#include "paddle/fluid/framework/details/multi_devices_graph_builder.h"
#include "paddle/fluid/framework/details/ssa_graph_printer.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/framework/details/fuse_all_reduce_graph_builder.h"
#endif

DEFINE_string(debug_ssa_graphviz_filename, "",
              "If this field is set, the graphviz format of ssa graph will be "
              "written to this file");

namespace paddle {
namespace framework {
namespace details {
std::unique_ptr<SSAGraphBuilder> SSAGraphBuilderFactory::Create() {
  std::unique_ptr<SSAGraphBuilder> res(
#ifdef PADDLE_WITH_CUDA
      new MultiDevSSAGraphBuilder(places_, loss_var_name_, param_names_,
                                  local_scopes_, nccl_ctxs_, strategy_)
#else
      new MultiDevSSAGraphBuilder(places_, loss_var_name_, param_names_,
                                  local_scopes_, strategy_)
#endif
          );  // NOLINT

  if (strategy_.reduce_ == BuildStrategy::ReduceStrategy::kFusedAllReduce) {
#ifdef PADDLE_WITH_CUDA
    res.reset(new FuseAllReduceGraphBuilder(std::move(res)));
#else
    PADDLE_THROW("Fused All Reduce Strategy is not supported in CPU.");
#endif
  }

  if (!FLAGS_debug_ssa_graphviz_filename.empty()) {
    std::ofstream fout(FLAGS_debug_ssa_graphviz_filename);
    PADDLE_ENFORCE(fout.good());

    std::unique_ptr<GraphvizSSAGraphPrinter> graphviz_printer(
        new GraphvizSSAGraphPrinter());
    res.reset(new SSAGraghBuilderWithPrinter(fout, std::move(graphviz_printer),
                                             std::move(res)));
  }
  return res;
}
}  // namespace details
}  // namespace framework
}  // namespace paddle
