#include "glog/logging.h"
#include "paddle/fluid/eager/function_api.h"
#include "paddle/tcmpt/api/all.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/eager/generated/nodes/sigmoid_node.h"

std::vector<std::vector<pt::Tensor>> GradNodesigmoid::operator()(const std::vector<std::vector<pt::Tensor>>& grads) {

  const std::shared_ptr<paddle::imperative::Tracer>& tracer = paddle::imperative::GetCurrentTracer();

  std::map<std::string, std::vector<std::shared_ptr<paddle::imperative::VarBase>>> ins = { { "Out", TensorsToVarBases(this->Out_) },{ "Out@GRAD", TensorsToVarBases(grads[0]) } };
  std::map<std::string, std::vector<std::shared_ptr<paddle::imperative::VarBase>>> outs = { { "X@GRAD", ConstructDuplicableOutput( this->OutputMeta()[0].Size() ) } };

  paddle::framework::AttributeMap attrs = { { "use_cudnn", this->use_cudnn_},{ "use_mkldnn", this->use_mkldnn_} };

  tracer->TraceOp("sigmoid_grad", ins, outs, attrs, tracer->ExpectedPlace(), false, {});

  std::vector<std::vector<pt::Tensor>> outputs(outs.size());
  outputs[0] = VarBasesToTensors(outs["X@GRAD"]);

  return outputs;
}