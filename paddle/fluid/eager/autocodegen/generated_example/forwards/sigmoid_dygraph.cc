#include "paddle/fluid/eager/generated/dygraph_forward_api.h"
#include "paddle/fluid/eager/generated/nodes/sigmoid_node.h"

pt::Tensor sigmoid_dygraph_function(const pt::Tensor& X, const bool use_mkldnn, const bool use_cudnn, const int op_role, const std::vector<std::string>& op_role_var, const std::string& op_namescope, const std::vector<std::string>& op_callstack, const std::string& op_device, const bool with_quant_attr, bool trace_backward) {

  // Dygraph Forward Pass
  const std::shared_ptr<paddle::imperative::Tracer>& tracer = paddle::imperative::GetCurrentTracer();

  std::map<std::string, std::vector<std::shared_ptr<paddle::imperative::VarBase>>> ins = { { "X", TensorsToVarBases(X) } };

  paddle::framework::AttributeMap attrs = { { "use_mkldnn", use_mkldnn }, { "use_cudnn", use_cudnn }, { "op_role", op_role }, { "op_role_var", op_role_var }, { "op_namescope", op_namescope }, { "op_callstack", op_callstack }, { "op_device", op_device }, { "with_quant_attr", with_quant_attr },  };

  std::map<std::string, std::vector<std::shared_ptr<paddle::imperative::VarBase>>> outs = { { "Out", ConstructDuplicableOutput(1) } };

  tracer->TraceOp("sigmoid", ins, outs, attrs, tracer->ExpectedPlace(), false, {});

  pt::Tensor Out = VarBasesToTensors(outs["Out"])[0];

  // Prepare Autograd Meta 
  egr::AutogradMeta& p_autograd_X = *egr::EagerUtils::unsafe_autograd_meta(X);
  egr::AutogradMeta& p_autograd_Out = *egr::EagerUtils::autograd_meta(&Out);

  egr::AutogradMeta* p_autograd_in = &p_autograd_X;
  egr::AutogradMeta** pp_autograd_in = &p_autograd_in;

  egr::AutogradMeta* p_autograd_out = &p_autograd_Out;
  egr::AutogradMeta** pp_autograd_out = &p_autograd_out;

  if(egr::EagerUtils::ComputeRequireGrad(pp_autograd_in, 1, pp_autograd_out, 1, trace_backward)) {
    // Create GradOpNode
    auto grad_node = std::make_shared<GradNodesigmoid>(1, 1);

    // Set Attributes
    grad_node->SetAttruse_cudnn(use_cudnn);
    grad_node->SetAttruse_mkldnn(use_mkldnn);

    // Set Tensor Wrappers
    grad_node->SetTensorWrapperOut(Out);

    grad_node->SetGradOutMeta(p_autograd_X, 0);
    grad_node->AddEdges(p_autograd_X, 0);
    grad_node->SetGradInMeta(p_autograd_Out, 0);
    egr::EagerUtils::SetOutRankWithSlot(&p_autograd_Out, 0);
    egr::EagerUtils::SetHistory(&p_autograd_Out, grad_node);

  }

  return Out;

}

USE_OP(sigmoid);
