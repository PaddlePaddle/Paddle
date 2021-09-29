#include "paddle/fluid/eager/generated/dygraph_forward_api.h"
#include "paddle/fluid/eager/generated/nodes/matmul_v2_node.h"

pt::Tensor matmul_v2_dygraph_function(const pt::Tensor& X,const pt::Tensor& Y, const bool trans_x, const bool trans_y, const bool use_mkldnn, const std::string& mkldnn_data_type, const int op_role, const std::vector<std::string>& op_role_var, const std::string& op_namescope, const std::vector<std::string>& op_callstack, const std::string& op_device, const bool with_quant_attr, bool trace_backward) {

  // Dygraph Forward Pass
  const std::shared_ptr<paddle::imperative::Tracer>& tracer = paddle::imperative::GetCurrentTracer();

  std::map<std::string, std::vector<std::shared_ptr<paddle::imperative::VarBase>>> ins = { { "X", TensorsToVarBases(X) },{ "Y", TensorsToVarBases(Y) } };

  paddle::framework::AttributeMap attrs = { { "trans_x", trans_x }, { "trans_y", trans_y }, { "use_mkldnn", use_mkldnn }, { "mkldnn_data_type", mkldnn_data_type }, { "op_role", op_role }, { "op_role_var", op_role_var }, { "op_namescope", op_namescope }, { "op_callstack", op_callstack }, { "op_device", op_device }, { "with_quant_attr", with_quant_attr },  };

  std::map<std::string, std::vector<std::shared_ptr<paddle::imperative::VarBase>>> outs = { { "Out", ConstructDuplicableOutput(1) } };

  tracer->TraceOp("matmul_v2", ins, outs, attrs, tracer->ExpectedPlace(), false, {});

  pt::Tensor Out = VarBasesToTensors(outs["Out"])[0];

  // Prepare Autograd Meta 
  egr::AutogradMeta& p_autograd_X = *egr::EagerUtils::unsafe_autograd_meta(X);
  egr::AutogradMeta& p_autograd_Y = *egr::EagerUtils::unsafe_autograd_meta(Y);
  egr::AutogradMeta& p_autograd_Out = *egr::EagerUtils::autograd_meta(&Out);

  std::vector<egr::AutogradMeta*> p_autograd_in;
  p_autograd_in.push_back(&p_autograd_X);
  p_autograd_in.push_back(&p_autograd_Y);
  egr::AutogradMeta** pp_autograd_in = p_autograd_in.data();

  egr::AutogradMeta* p_autograd_out = &p_autograd_Out;
  egr::AutogradMeta** pp_autograd_out = &p_autograd_out;

  if(egr::EagerUtils::ComputeRequireGrad(pp_autograd_in, p_autograd_in.size(), pp_autograd_out, 1, trace_backward)) {
    // Create GradOpNode
    auto grad_node = std::make_shared<GradNodematmul_v2>(1, 2);

    // Set Attributes
    grad_node->SetAttrmkldnn_data_type(mkldnn_data_type);
    grad_node->SetAttruse_mkldnn(use_mkldnn);
    grad_node->SetAttrtrans_x(trans_x);
    grad_node->SetAttrtrans_y(trans_y);

    // Set Tensor Wrappers
    grad_node->SetTensorWrapperX(X);
    grad_node->SetTensorWrapperY(Y);

    grad_node->SetGradOutMeta(p_autograd_X, 0);
    grad_node->AddEdges(p_autograd_X, 0);
    grad_node->SetGradOutMeta(p_autograd_Y, 1);
    grad_node->AddEdges(p_autograd_Y, 1);
    grad_node->SetGradInMeta(p_autograd_Out, 0);
    egr::EagerUtils::SetOutRankWithSlot(&p_autograd_Out, 0);
    egr::EagerUtils::SetHistory(&p_autograd_Out, grad_node);

  }

  return Out;

}

USE_OP(matmul_v2);
