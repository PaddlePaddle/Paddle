#pragma once

#include <boost/variant.hpp>
#include <cstddef>
#include <memory>
#include <vector>
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace inference {
namespace anakin {

enum DataType { kUnk = -1, kFloat32, kFloat64, kInt32 };
enum Place { kCpu = 0, kGpu };

using shape_t = std::vector<int>;
using attr_t = boost::variant<int, bool>;

struct Tensor {
  DataType dtype{DataType::kUnk};
  void* buffer{nullptr};
  size_t buffer_size{0};
  shape_t shape;
  Place place{Place::kCpu};
  std::string name;  // for specifying inputs.
};

class AnakinEngine {
 public:
  // ir::Node also has OpDesc.
  static bool IsOpSupported(framework::ir::Node* node);

  // Topology releated.
  void DeclareInput(const std::string& id, DataType dtype,
                    const shape_t& shape);
  void DeclareOutput(const std::string& id);

  void AddOp(const std::string& op_type, const std::vector<std::string>& inputs,
             const std::vector<std::string>& outputs,
             const std::vector<attr_t>& attrs);
  void AddVar(const std::string& id, DataType dtype, const shape_t& shape);
  Tensor* AddWeight(const std::string& id, DataType dtype,
                    const shape_t& shape);

  void FreezeNetwork();

  void Execute(const std::vector<Tensor>& inputs,
               const std::vector<Tensor>* outputs, int batch_size = 0);

  // Runtime
  bool SetInput(const std::string& id, const Tensor& data);
  bool GetOutput(const std::string& id, Tensor* data);
  bool GetVar(const std::string& id, Tensor* data);

 private:
  void* raw_engine_{nullptr};
};

// Pseudo codes here!
TEST(test, main) {
  // in anakin_subgraph_analysis_pass
  // ir::Graph graph;

  declare subgraph;
  for (auto* node : graph) {
    if (AnakinEngine::IsOpSupported(node)) {
      // add to subgraph
    }
  }

  if (IsSubgraphValid(subgraph)) {
    // build the engine
    AnakinEngine engine;
    std::unordered_set<std::string> collected_vars;

    auto inputs = CollectSubgraphInputs(subgraph);
    auto outputs = CollectSubgraphOutputs(subgraph);

    // Declare subgraph inputs and outputs
    for (auto x : inputs) {
      engine.DeclareInput(x->name, x->dtype, x->shape);
    }
    for (auto x : outputs) {
      engine.DeclareOutput(x->name);
    }

    // Build the subgraph network.
    for (auto* node : subgraph) {
      // create vars and weights.
      for (auto* input : node->inputs) {
        if (collected_vars.count(input.name)) continue;
        if (input.IsWeight()) {
          engine.AddWeight(node.name, input.dtype, input.shape);
        } else {
          engine.AddVar(node.name, input.dtype, input.shape);
        }
      }

      for (auto* output : node->output) {
        if (collected_vars.count(input.name)) continue;
        engine.AddVar(node.name, input.dtype, input.shape);
      }

      engine.AddOp(node->op_type, node->inputs, node->outputs, node->attrs);
    }

    engine.FreezeNetwork();

    // Run inference
    engine.Execute(...);
  }
}

}  // namespace tensorrt
}  // namespace inference
}  // namespace paddle
