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

#pragma once

#include <iosfwd>
#include <string>
#include "paddle/fluid/framework/details/ssa_graph_builder.h"

namespace paddle {
namespace framework {
namespace details {
struct SSAGraph;
class SSAGraphPrinter {
 public:
  virtual ~SSAGraphPrinter() {}
  virtual void Print(const SSAGraph& graph, std::ostream& sout) const = 0;
};

class GraphvizSSAGraphPrinter : public SSAGraphPrinter {
 public:
  void Print(const SSAGraph& graph, std::ostream& sout) const override;
};

class SSAGraghBuilderWithPrinter : public SSAGraphBuilder {
 public:
  SSAGraghBuilderWithPrinter(std::ostream& sout,
                             std::unique_ptr<SSAGraphPrinter>&& printer,
                             std::unique_ptr<SSAGraphBuilder>&& builder)
      : printer_(std::move(printer)),
        builder_(std::move(builder)),
        stream_ref_(sout) {}

  SSAGraghBuilderWithPrinter(std::unique_ptr<std::ostream>&& sout,
                             std::unique_ptr<SSAGraphPrinter>&& printer,
                             std::unique_ptr<SSAGraphBuilder>&& builder)
      : printer_(std::move(printer)),
        builder_(std::move(builder)),
        stream_ptr_(std::move(sout)),
        stream_ref_(*stream_ptr_) {}

  std::unique_ptr<SSAGraph> Build(const ProgramDesc& program) const override {
    auto graph = builder_->Build(program);
    printer_->Print(*graph, stream_ref_);
    return graph;
  }

  int GetVarDeviceID(const std::string& var_name) const override {
    return builder_->GetVarDeviceID(var_name);
  }

 private:
  std::unique_ptr<SSAGraphPrinter> printer_;
  std::unique_ptr<SSAGraphBuilder> builder_;
  std::unique_ptr<std::ostream> stream_ptr_;
  std::ostream& stream_ref_;
};

}  // namespace details
}  // namespace framework
}  // namespace paddle
