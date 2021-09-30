#include "glog/logging.h"
#include "paddle/fluid/eager/function_api.h"
#include "paddle/tcmpt/api/all.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/eager/generated/nodes/matmul_v2_node.h"

std::vector<std::vector<pt::Tensor>> GradNodematmul_v2::operator()(const std::vector<std::vector<pt::Tensor>>& grads) {

  const std::shared_ptr<paddle::imperative::Tracer>& tracer = paddle::imperative::GetCurrentTracer();

  std::map<std::string, std::vector<std::shared_ptr<paddle::imperative::VarBase>>> ins = { { "Out@GRAD", TensorsToVarBases(grads[0]) },{ "X", TensorsToVarBases(this->X_) },{ "Y", TensorsToVarBases(this->Y_) } };
  std::map<std::string, std::vector<std::shared_ptr<paddle::imperative::VarBase>>> outs = { { "X@GRAD", ConstructDuplicableOutput( this->OutputMeta()[0].Size() ) },{ "Y@GRAD", ConstructDuplicableOutput( this->OutputMeta()[1].Size() ) } };

  paddle::framework::AttributeMap attrs = { { "mkldnn_data_type", this->mkldnn_data_type_},{ "use_mkldnn", this->use_mkldnn_},{ "trans_x", this->trans_x_},{ "trans_y", this->trans_y_} };

  tracer->TraceOp("matmul_v2_grad", ins, outs, attrs, tracer->ExpectedPlace(), false, {});

  std::vector<std::vector<pt::Tensor>> outputs(outs.size());
  outputs[0] = VarBasesToTensors(outs["X@GRAD"]);
  outputs[1] = VarBasesToTensors(outs["Y@GRAD"]);

  return outputs;
}