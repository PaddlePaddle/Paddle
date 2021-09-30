#include "glog/logging.h"
#include "paddle/fluid/eager/autograd_meta.h"
#include "paddle/tcmpt/api/all.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/eager/utils.h"
#include "paddle/fluid/framework/op_registry.h"

pt::Tensor matmul_v2_dygraph_function(const pt::Tensor& X,const pt::Tensor& Y, const bool trans_x, const bool trans_y, const bool use_mkldnn, const std::string& mkldnn_data_type, const int op_role, const std::vector<std::string>& op_role_var, const std::string& op_namescope, const std::vector<std::string>& op_callstack, const std::string& op_device, const bool with_quant_attr, bool trace_backward);
pt::Tensor sigmoid_dygraph_function(const pt::Tensor& X, const bool use_mkldnn, const bool use_cudnn, const int op_role, const std::vector<std::string>& op_role_var, const std::string& op_namescope, const std::vector<std::string>& op_callstack, const std::string& op_device, const bool with_quant_attr, bool trace_backward);
