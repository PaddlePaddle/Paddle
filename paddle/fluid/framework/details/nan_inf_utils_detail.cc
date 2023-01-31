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

#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#endif
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/phi/kernels/funcs/eigen/extensions.h"

DECLARE_int32(check_nan_inf_level);

namespace paddle {
namespace framework {
namespace details {

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

template <
    typename T,
    std::enable_if_t<!std::is_same<T, phi::dtype::complex<float>>::value &&
                         !std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
static void CheckNanInfCpuImpl(const T* value_ptr,
                               const int64_t numel,
                               const std::string& cpu_hint_str) {
  using MT = typename phi::dtype::template MPTypeTrait<T>::Type;

#ifdef _OPENMP
  // Use maximum 4 threads to collect the nan and inf information.
  int num_threads = std::max(omp_get_num_threads(), 1);
  num_threads = std::min(num_threads, 4);
#else
  int num_threads = 1;
#endif

  std::vector<int64_t> thread_num_nan(num_threads, 0);
  std::vector<int64_t> thread_num_inf(num_threads, 0);
  std::vector<MT> thread_min_value(num_threads, static_cast<MT>(value_ptr[0]));
  std::vector<MT> thread_max_value(num_threads, static_cast<MT>(value_ptr[0]));
  std::vector<MT> thread_mean_value(num_threads, static_cast<MT>(0));

#ifdef _OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
  {
#ifdef _OPENMP
    int64_t tid = omp_get_thread_num();
    int64_t chunk_size = (numel + num_threads - 1) / num_threads;
    int64_t begin = tid * chunk_size;
    int64_t end = chunk_size + begin > numel ? numel : chunk_size + begin;
#else
    int64_t tid = 0;
    int64_t begin = 0;
    int64_t end = numel;
#endif
    for (int64_t i = begin; i < end; ++i) {
      MT value = static_cast<MT>(value_ptr[i]);

      thread_min_value[tid] = std::min(thread_min_value[tid], value);
      thread_max_value[tid] = std::max(thread_max_value[tid], value);
      thread_mean_value[tid] += value / static_cast<MT>(numel);

      if (std::isnan(value)) {
        thread_num_nan[tid] += 1;
      } else if (std::isinf(value)) {
        thread_num_inf[tid] += 1;
      }
    }
  }

  int64_t num_nan = 0;
  int64_t num_inf = 0;
  MT min_value = thread_min_value[0];
  MT max_value = thread_max_value[0];
  MT mean_value = static_cast<MT>(0);
  for (int i = 0; i < num_threads; ++i) {
    num_nan += thread_num_nan[i];
    num_inf += thread_num_inf[i];
    min_value = std::min(thread_min_value[i], min_value);
    max_value = std::max(thread_max_value[i], max_value);
    mean_value += thread_mean_value[i];
  }

  PrintForDifferentLevel<T, MT>(cpu_hint_str.c_str(),
                                numel,
                                num_nan,
                                num_inf,
                                max_value,
                                min_value,
                                mean_value,
                                FLAGS_check_nan_inf_level);
}

template <
    typename T,
    std::enable_if_t<std::is_same<T, phi::dtype::complex<float>>::value ||
                         std::is_same<T, phi::dtype::complex<double>>::value,
                     bool> = true>
void CheckNanInfCpuImpl(const T* value_ptr,
                        const int64_t numel,
                        const std::string& cpu_hint_str) {
  using RealType = typename T::value_type;

  RealType real_sum = 0.0f, imag_sum = 0.0f;

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : real_sum) reduction(+ : imag_sum)
#endif
  for (int64_t i = 0; i < numel; ++i) {
    T value = value_ptr[i];
    real_sum += (value.real - value.real);
    imag_sum += (value.imag - value.imag);
  }

  if (std::isnan(real_sum) || std::isinf(real_sum) || std::isnan(imag_sum) ||
      std::isinf(imag_sum)) {
    // hot fix for compile failed in gcc4.8
    // here also need print detail info of nan or inf later
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "There are NAN or INF in %s.", cpu_hint_str));
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
  } else if (platform::is_npu_place(tensor->place())) {
#ifdef PADDLE_WITH_ASCEND_CL
    if (framework::TransToProtoVarType(tensor->dtype()) !=
        proto::VarType::FP32) {
      return;
    }

    phi::DenseTensor cpu_tensor;
    cpu_tensor.Resize(tensor->dims());
    float* cpu_data = static_cast<float*>(
        cpu_tensor.mutable_data(platform::CPUPlace(), tensor->dtype()));

    framework::TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
    bool flag = false;
    for (int i = 0; i < cpu_tensor.numel(); i++) {
      if (isnan(cpu_data[i]) || isinf(cpu_data[i])) {
        flag = true;
        break;
      }
    }
    PADDLE_ENFORCE_NE(
        flag,
        true,
        platform::errors::Fatal(
            "Operator %s output phi::DenseTensor %s contains Inf.",
            op_type,
            var_name));
#else
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "phi::DenseTensor[%s] use npu place. PaddlePaddle must compile "
        "with NPU.",
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

#ifdef PADDLE_WITH_ASCEND_CL
using NpuOpRunner = paddle::operators::NpuOpRunner;

constexpr int FLOAT_STATUS_SIZE = 8;

static phi::DenseTensor& npu_float_status() {
  static phi::DenseTensor float_status;
  return float_status;
}

void NPUAllocAndClearFloatStatus(const framework::OperatorBase& op,
                                 const framework::Scope& scope,
                                 const platform::Place& place) {
  if (!platform::is_npu_place(place)) return;

  std::call_once(white_list_init_flag, InitWhiteListFormEnv);
  if (IsSkipOp(op)) return;

  auto* dev_ctx = reinterpret_cast<platform::NPUDeviceContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  auto stream = dev_ctx->stream();

  auto& flag = npu_float_status();
  flag.mutable_data<float>({FLOAT_STATUS_SIZE}, place);
  NpuOpRunner("NPUAllocFloatStatus", {}, {flag}).Run(stream);

  phi::DenseTensor tmp;
  tmp.mutable_data<float>({FLOAT_STATUS_SIZE}, place);
  NpuOpRunner("NPUClearFloatStatus", {tmp}, {flag}).Run(stream);
}

void PrintNpuVarInfo(const std::string& op_type,
                     const std::string& var_name,
                     const framework::Variable* var,
                     const platform::Place& place) {
  const phi::DenseTensor* tensor{nullptr};
  if (var->IsType<phi::DenseTensor>()) {
    tensor = &var->Get<phi::DenseTensor>();
  } else if (var->IsType<phi::SelectedRows>()) {
    tensor = &var->Get<phi::SelectedRows>().value();
  } else {
    VLOG(10) << var_name << " var_name need not to check";
    return;
  }

  if ((framework::TransToProtoVarType(tensor->dtype()) !=
       proto::VarType::FP32) &&
      (framework::TransToProtoVarType(tensor->dtype()) !=
       proto::VarType::FP16)) {
    return;
  }

  if (tensor->memory_size() == 0) {
    VLOG(10) << var_name << " var_name need not to check, size == 0";
    return;
  }

  VLOG(10) << "begin check " << op_type << " var_name:" << var_name
           << ", place:" << tensor->place() << ", numel:" << tensor->numel();

  phi::DenseTensor cpu_tensor;
  cpu_tensor.Resize(tensor->dims());
  cpu_tensor.mutable_data(platform::CPUPlace(), tensor->dtype());
  framework::TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);

  LOG(WARNING) << "print [" << var_name << "] tensor info:";
  // use env strategy control in future, -1=print_all.
  int print_num = 3;
  if (framework::TransToProtoVarType(tensor->dtype()) == proto::VarType::FP32) {
    const float* value = cpu_tensor.data<float>();
    PrintNanInf(value, tensor->numel(), print_num, op_type, var_name, false);
  } else if (framework::TransToProtoVarType(tensor->dtype()) ==
             proto::VarType::FP16) {
    const paddle::platform::float16* value =
        cpu_tensor.data<paddle::platform::float16>();
    PrintNanInf(value, tensor->numel(), print_num, op_type, var_name, false);
  }
}

void PrintNPUOpValueInfo(const framework::OperatorBase& op,
                         const framework::Scope& scope,
                         const platform::Place& place) {
  LOG(WARNING) << "There are `nan` or `inf` in operator (" << op.Type()
               << "), here we print some tensor value info of this op.";
  for (auto& vname : op.InputVars()) {
    auto* var = scope.FindVar(vname);
    if (var == nullptr) continue;
    PrintNpuVarInfo(op.Type(), vname, var, place);
  }

  for (auto& vname : op.OutputVars(true)) {
    auto* var = scope.FindVar(vname);
    if (var == nullptr) continue;
    PrintNpuVarInfo(op.Type(), vname, var, place);
  }
}

static void NPUCheckOpHasNanOrInf(const framework::OperatorBase& op,
                                  const framework::Scope& scope,
                                  const platform::Place& place) {
  if (!platform::is_npu_place(place)) return;

  auto* dev_ctx = reinterpret_cast<platform::NPUDeviceContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  auto stream = dev_ctx->stream();

  auto& flag = npu_float_status();
  phi::DenseTensor tmp;
  tmp.mutable_data<float>({FLOAT_STATUS_SIZE}, place);
  // NPUGetFloatStatus updates data on input in-place.
  // tmp is only placeholder.
  NpuOpRunner("NPUGetFloatStatus", {flag}, {tmp}).Run(stream);

  phi::DenseTensor cpu_tensor;
  auto cpu_place = platform::CPUPlace();
  float* cpu_data = static_cast<float*>(
      cpu_tensor.mutable_data<float>({FLOAT_STATUS_SIZE}, cpu_place));

  framework::TensorCopySync(flag, cpu_place, &cpu_tensor);
  float sum = 0.0;
  for (int i = 0; i < FLOAT_STATUS_SIZE; ++i) {
    sum += cpu_data[i];
  }

  if (sum >= 1.0) PrintNPUOpValueInfo(op, scope, place);

  PADDLE_ENFORCE_LT(sum,
                    1.0,
                    platform::errors::PreconditionNotMet(
                        "Operator %s contains Nan/Inf.", op.Type()));
}
#endif

void CheckOpHasNanOrInf(const framework::OperatorBase& op,
                        const framework::Scope& exec_scope,
                        const platform::Place& place) {
  std::call_once(white_list_init_flag, InitWhiteListFormEnv);

  if (IsSkipOp(op)) return;

#ifdef PADDLE_WITH_ASCEND_CL
  if (platform::is_npu_place(place)) {
    NPUCheckOpHasNanOrInf(op, exec_scope, place);
    return;
  }
#endif

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
