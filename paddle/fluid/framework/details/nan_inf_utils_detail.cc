// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/nan_inf_utils_detail.h"

#include "paddle/fluid/framework/details/nan_inf_utils.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/phi/common/amp_type_traits.h"

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/phi/core/flags.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"

PHI_DECLARE_int32(check_nan_inf_level);

namespace paddle {
namespace framework {
namespace details {
struct DebugTools {
  DebugTools() {}
  std::string path = "";
  int stack_limit = 1;
};
static DebugTools debug_nan_inf;

void SetNanInfDebugPath(const std::string& nan_inf_path) {
  debug_nan_inf.path = nan_inf_path;
  VLOG(4) << "Set the log's path of debug tools : " << nan_inf_path;
}

std::string GetNanPath() {
  if (debug_nan_inf.path.empty()) {
    return "";
  }
  return debug_nan_inf.path + "/";
}

void SetNanInfStackLimit(const int& stack_limit) {
  debug_nan_inf.stack_limit = stack_limit;
  VLOG(4) << "Set the stack limit of debug tools : " << stack_limit;
}

int GetNanInfStackLimit() { return debug_nan_inf.stack_limit; }

static std::once_flag white_list_init_flag;

static int op_role_nan_inf_white_list = 0;

static constexpr int FORWARD = 0x10000;

// lazy init
static const std::unordered_map<std::string, int>& role_str2int() {
  /* In op_proto_maker.h
   * framework::OpRole::kForward      = 0x0000,
   * framework::OpRole::kBackward     = 0x0001,
   * framework::OpRole::kOptimize     = 0x0002,
   * framework::OpRole::kRPC          = 0x0004,
   * framework::OpRole::kDist         = 0x0008,
   * framework::OpRole::kLRSched      = 0x0010,
   * framework::OpRole::kLoss         = 0x0100,
   * framework::OpRole::kNotSpecified = 0x1000,
   */
  static const std::unordered_map<std::string, int> _role_str2int = {
      {"forward", FORWARD}, /* kForward=0, can't filter */
      {"backward", static_cast<int>(framework::OpRole::kBackward)},
      {"optimize", static_cast<int>(framework::OpRole::kOptimize)},
      {"rpc", static_cast<int>(framework::OpRole::kRPC)},
      {"dist", static_cast<int>(framework::OpRole::kDist)},
      {"lrsched", static_cast<int>(framework::OpRole::kLRSched)},
      {"loss", static_cast<int>(framework::OpRole::kLoss)},
      {"default", static_cast<int>(framework::OpRole::kNotSpecified)},
  };
  return _role_str2int;
}

static std::unordered_set<std::string>& op_type_nan_inf_white_list() {
  static std::unordered_set<std::string> _op_type_nan_inf_white_list = {
      "coalesce_tensor", /* This Op will alloc tensor, and may not init space */
  };
  return _op_type_nan_inf_white_list;
}

static std::unordered_map<std::string, std::vector<std::string>>&
op_var_nan_inf_white_list() {
  static std::unordered_map<std::string, std::vector<std::string>>
      _op_var_nan_inf_white_list = {
          /* encoded & gather var consist of idx&val, can't judge directly */
          {"dgc", {"__dgc_encoded__", "__dgc_gather__"}},
      };
  return _op_var_nan_inf_white_list;
}

static void InitWhiteListFormEnv() {
  // op_type_skip and op_var_skip may be NULL.
  // So need init static value in there, prevent thread competition.
  // NOTE. role_str2int needn't do this for it only used in this func.
  op_type_nan_inf_white_list();
  op_var_nan_inf_white_list();

  // export PADDLE_INF_NAN_SKIP_OP="op0,op1,op2"
  // export PADDLE_INF_NAN_SKIP_ROLE="role1,role2,role3"
  // export PADDLE_INF_NAN_SKIP_VAR="op0:var0,op0:var1,op1:var0"
  const char* op_type_skip = std::getenv("PADDLE_INF_NAN_SKIP_OP");
  const char* op_role_skip = std::getenv("PADDLE_INF_NAN_SKIP_ROLE");
  const char* op_var_skip = std::getenv("PADDLE_INF_NAN_SKIP_VAR");

  if (op_type_skip) {
    std::stringstream ss(op_type_skip);
    std::string op_type;
    while (std::getline(ss, op_type, ',')) {
      op_type_nan_inf_white_list().emplace(op_type);
    }
  }

  if (op_role_skip) {
    std::stringstream ss(op_role_skip);
    std::string op_role;
    while (std::getline(ss, op_role, ',')) {
      PADDLE_ENFORCE_EQ(role_str2int().find(op_role) != role_str2int().end(),
                        true,
                        platform::errors::InvalidArgument(
                            "Skip role must be one of "
                            "{forward,backward,optimize,rpc,dist,lrsched,loss,"
                            "default}, instead of %s",
                            op_role));
      op_role_nan_inf_white_list |= role_str2int().at(op_role);
    }
  }

  if (op_var_skip) {
    std::stringstream ss(op_var_skip);
    std::string op_var;
    while (std::getline(ss, op_var, ',')) {
      auto pos = op_var.find(":");
      PADDLE_ENFORCE_EQ(
          pos != std::string::npos,
          true,
          platform::errors::InvalidArgument(
              "Skip var format must be op:var, instead of %s", op_var));
      std::string op = op_var.substr(0, pos);
      std::string var = op_var.substr(pos + 1);

      op_var_nan_inf_white_list()[op].emplace_back(var);
    }
  }
}

template <>
template <typename T>
void TensorCheckerVisitor<phi::CPUContext>::apply(
    typename std::enable_if<
        std::is_floating_point<T>::value ||
        std::is_same<T, ::paddle::platform::complex<float>>::value ||
        std::is_same<T, ::paddle::platform::complex<double>>::value>::type*)
    const {
  std::string cpu_hint_str =
      GetCpuHintString<T>(op_type, var_name, tensor.place());
  CheckNanInfCpuImpl(tensor.data<T>(), tensor.numel(), cpu_hint_str);
}

template <>
void tensor_check<phi::CPUContext>(const std::string& op_type,
                                   const std::string& var_name,
                                   const phi::DenseTensor& tensor,
                                   const platform::Place& place) {
  TensorCheckerVisitor<phi::CPUContext> vistor(
      op_type, var_name, tensor, place);
  VisitDataType(framework::TransToProtoVarType(tensor.dtype()), vistor);
}

#ifdef PADDLE_WITH_XPU

template <typename T, typename Dummy = void>
struct XPUCheckNanOrInfImpl;

template <typename T>
struct XPUCheckNanOrInfImpl<
    T,
    typename std::enable_if<
        std::is_integral<T>::value ||
        std::is_same<T, ::paddle::platform::complex<float>>::value ||
        std::is_same<T, ::paddle::platform::complex<double>>::value ||
        std::is_same<T, ::paddle::platform::bfloat16>::value ||
        std::is_same<T, double>::value>::type> {
  XPUCheckNanOrInfImpl(const std::string& o, const std::string& v)
      : op_type(o), var_name(v) {}

  bool operator()(const phi::XPUContext&,
                  const phi::DenseTensor& has_nan_or_inf) const {
    VLOG(10) << var_name << " need not to check, it's type is not float point";
    return false;
  }

  std::string op_type;
  std::string var_name;
};

template <typename T>
struct XPUCheckNanOrInfImpl<
    T,
    typename std::enable_if<
        std::is_same<T, float>::value ||
        std::is_same<T, ::paddle::platform::float16>::value>::type> {
  XPUCheckNanOrInfImpl(const std::string& o, const std::string& v)
      : op_type(o), var_name(v) {}

  bool operator()(const phi::XPUContext& dev_ctx,
                  const phi::DenseTensor& tensor) const {
    using XPUType = typename XPUTypeTrait<T>::Type;

    phi::DenseTensor has_nan_or_inf;
    has_nan_or_inf.Resize({1});
    dev_ctx.template Alloc<bool>(&has_nan_or_inf);
    xpu::check_nan_or_inf(dev_ctx.x_context(),
                          reinterpret_cast<const XPUType*>(tensor.data()),
                          reinterpret_cast<typename XPUTypeTrait<bool>::Type*>(
                              has_nan_or_inf.data()),
                          tensor.numel());

    phi::DenseTensor cpu_tensor;
    paddle::platform::CPUPlace cpu_place;
    cpu_tensor.Resize({1});
    paddle::platform::DeviceContextPool::Instance()
        .Get(cpu_place)
        ->template Alloc<bool>(&cpu_tensor);
    paddle::memory::Copy(cpu_place,
                         cpu_tensor.data(),
                         has_nan_or_inf.place(),
                         has_nan_or_inf.data(),
                         paddle::experimental::SizeOf(cpu_tensor.dtype()));
    dev_ctx.Wait();
    return *cpu_tensor.data<bool>();
  }

  std::string op_type;
  std::string var_name;
};

template <>
template <typename T>
void TensorCheckerVisitor<phi::XPUContext>::apply(
    typename std::enable_if<
        std::is_floating_point<T>::value ||
        std::is_same<T, ::paddle::platform::complex<float>>::value ||
        std::is_same<T, ::paddle::platform::complex<double>>::value>::type*)
    const {
  XPUCheckNanOrInfImpl<T> check_nan_or_inf_impl(op_type, var_name);
  auto* dev_ctx = reinterpret_cast<phi::XPUContext*>(
      platform::DeviceContextPool::Instance().Get(tensor.place()));
  if (check_nan_or_inf_impl(*dev_ctx, tensor)) {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "There are NAN or INF in %s, %s.", op_type, var_name));
  }
}

template <>
void tensor_check<phi::XPUContext>(const std::string& op_type,
                                   const std::string& var_name,
                                   const phi::DenseTensor& tensor,
                                   const platform::Place& place) {
  TensorCheckerVisitor<phi::XPUContext> vistor(
      op_type, var_name, tensor, place);
  VisitDataType(framework::TransToProtoVarType(tensor.dtype()), vistor);
}

#endif

void CheckVarHasNanOrInf(const std::string& op_type,
                         const std::string& var_name,
                         const framework::Variable* var,
                         const platform::Place& place) {
  PADDLE_ENFORCE_NOT_NULL(
      var,
      platform::errors::NotFound(
          "Cannot find var: `%s` in op `%s`.", var_name, op_type));

  const phi::DenseTensor* tensor{nullptr};
  if (var->IsType<phi::DenseTensor>()) {
    tensor = &var->Get<phi::DenseTensor>();
  } else if (var->IsType<phi::SelectedRows>()) {
    tensor = &var->Get<phi::SelectedRows>().value();
  } else {
    VLOG(10) << var_name << " var_name need not to check";
    return;
  }

  if (tensor->memory_size() == 0) {
    VLOG(10) << var_name << " var_name need not to check, size == 0";
    return;
  }

  VLOG(10) << "begin check " << op_type << " var_name:" << var_name
           << ", place:" << tensor->place() << ", numel:" << tensor->numel();

  if (platform::is_gpu_place(tensor->place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    tensor_check<phi::GPUContext>(op_type, var_name, *tensor, place);
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "phi::DenseTensor[%s] use gpu place. PaddlePaddle must compile "
        "with GPU.",
        var_name));
#endif
    return;
  } else if (platform::is_xpu_place(tensor->place())) {
#ifdef PADDLE_WITH_XPU
    if (framework::TransToProtoVarType(tensor->dtype()) !=
        proto::VarType::FP32) {
      return;
    }

    float* cpu_data = new float[tensor->numel()];
    memory::Copy(platform::CPUPlace(),
                 static_cast<void*>(cpu_data),
                 tensor->place(),
                 static_cast<const void*>(tensor->data<float>()),
                 tensor->numel() * sizeof(float));
    bool flag = false;
    for (int i = 0; i < tensor->numel(); i++) {
      if (isnan(cpu_data[i]) || isinf(cpu_data[i])) {
        flag = true;
        break;
      }
    }
    delete[] cpu_data;
    PADDLE_ENFORCE_NE(
        flag,
        true,
        platform::errors::Fatal(
            "Operator %s output phi::DenseTensor %s contains Inf.",
            op_type,
            var_name));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "phi::DenseTensor[%s] use xpu place. PaddlePaddle must compile "
        "with XPU.",
        var_name));
#endif
    return;
  }
  tensor_check<phi::CPUContext>(op_type, var_name, *tensor, place);
}

void CheckVarHasNanOrInf(const std::string& op_type,
                         const framework::Scope& scope,
                         const std::string& var_name,
                         const platform::Place& place) {
  auto* var = scope.FindVar(var_name);
  CheckVarHasNanOrInf(op_type, var_name, var, place);
}

bool IsSkipOp(const framework::OperatorBase& op) {
  if (op_type_nan_inf_white_list().count(op.Type()) != 0) return true;

  int op_role = 0;
  if (op.HasAttr(framework::OpProtoAndCheckerMaker::OpRoleAttrName())) {
    op_role = op.template Attr<int>(
        framework::OpProtoAndCheckerMaker::OpRoleAttrName());
  }

  // kForward=0, can't filter
  if (op_role == static_cast<int>(framework::OpRole::kForward)) {
    op_role = FORWARD;
  }
  if (op_role_nan_inf_white_list & op_role) return true;

  return false;
}

void CheckOpHasNanOrInf(const framework::OperatorBase& op,
                        const framework::Scope& exec_scope,
                        const platform::Place& place) {
  std::call_once(white_list_init_flag, InitWhiteListFormEnv);

  if (IsSkipOp(op)) return;

  if (op_var_nan_inf_white_list().count(op.Type()) == 0) {
    // NOTE. vname may destruct in the end of this func.
    for (auto& vname : op.OutputVars(true)) {
      auto* var = exec_scope.FindVar(vname);
      if (var == nullptr) continue;
      CheckVarHasNanOrInf(op.Type(), exec_scope, vname, place);
    }
  } else {
    for (auto& vname : op.OutputVars(true)) {
      bool need_check = true;
      for (auto& white_vname : op_var_nan_inf_white_list().at(op.Type())) {
        if (vname.find(white_vname) != std::string::npos) {
          need_check = false;
          break;
        }
      }
      if (!need_check) continue;
      auto* var = exec_scope.FindVar(vname);
      if (var == nullptr) continue;
      CheckVarHasNanOrInf(op.Type(), exec_scope, vname, place);
    }
  }
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
