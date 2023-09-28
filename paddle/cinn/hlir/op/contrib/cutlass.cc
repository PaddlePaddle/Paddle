// Copyright (c) 2023 CINN Authors. All Rights Reserved.
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

#include <Python.h>
#include "paddle/cinn/backends/codegen_cuda_util.h"
#include "paddle/cinn/hlir/framework/op_strategy.h"
#include "paddle/cinn/hlir/framework/tensor.h"

namespace cinn {
namespace hlir {
namespace op {

using common::_CINNValuePack_;
using common::CINNValue;
using common::CINNValuePack;
using framework::OpStrategy;
using framework::shape_t;
using framework::StrategyFunction;

using ArgsFunc =
    std::function<std::vector<ir::Expr>(const framework::NodeAttr &,
                                        const std::vector<ir::Tensor> &,
                                        const std::vector<std::vector<int>> &)>;

class CutlassCallArgsFuncRegistry {
 public:
  static CutlassCallArgsFuncRegistry &Global() {
    static CutlassCallArgsFuncRegistry instance;
    return instance;
  }

  void Register(const std::string &cutlass_op,
                const common::Target &target,
                ArgsFunc args_func) {
    auto id = cutlass_op + "_" + target.arch_str();
    func_map_[id] = args_func;
  }

  ArgsFunc Lookup(const std::string &cutlass_op, const common::Target &target) {
    auto id = cutlass_op + "_" + target.arch_str();
    CHECK(func_map_.count(id))
        << "Can't find " << cutlass_op << " for target " << target.arch_str();
    return func_map_[id];
  }

 private:
  CutlassCallArgsFuncRegistry() {}
  std::unordered_map<std::string, ArgsFunc> func_map_;
};

std::string GetCutlassKernel(const std::string &op_name,
                             const std::vector<ir::Tensor> &inputs,
                             const std::vector<Type> &out_type) {
  int sm = 80;
  int M = inputs[0]->shape[0].as_int32();
  int N = inputs[1]->shape[1].as_int32();
  int K = inputs[0]->shape[1].as_int32();
  std::string in0_dtype = common::Type2Str(inputs[0]->type());
  std::string in1_dtype = common::Type2Str(inputs[1]->type());
  std::string out_dtype = common::Type2Str(out_type[0]);
  bool find_first_valid = true;
  VLOG(4) << op_name;
  VLOG(4) << M;
  VLOG(4) << N;
  VLOG(4) << K;
  VLOG(4) << in0_dtype;
  VLOG(4) << in1_dtype;
  VLOG(4) << out_dtype;

  // 1. initial params
  PyObject *pModule = NULL;
  PyObject *pFunc = NULL;
  PyObject *pArgs = NULL;
  PyObject *pReturn = NULL;
  PyObject *objectsRepresentation = NULL;

  // 2. import module
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("sys.path.append('../../../../../python/cinn/')");
  pModule = PyImport_ImportModule("cutlass.gen_kernel");
  if (pModule == NULL) {
    PyErr_Print();
    return "";
  }

  // 3、get the func
  pFunc = PyObject_GetAttrString(pModule, "gen_gemm_kernel");
  if (pFunc == NULL) {
    PyErr_Print();
    return "";
  }

  // 4. set the args
  pArgs = PyTuple_New(9);

  PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", sm));
  PyTuple_SetItem(pArgs, 1, Py_BuildValue("s", op_name.c_str()));
  PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", M));
  PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", N));
  PyTuple_SetItem(pArgs, 4, Py_BuildValue("i", K));
  PyTuple_SetItem(pArgs, 5, Py_BuildValue("s", in0_dtype.c_str()));
  PyTuple_SetItem(pArgs, 6, Py_BuildValue("s", in1_dtype.c_str()));
  PyTuple_SetItem(pArgs, 7, Py_BuildValue("s", out_dtype.c_str()));
  // PyTuple_SetItem(pArgs, 8, Py_BuildValue("i",
  // static_cast<int>(find_first_valid)));
  PyTuple_SetItem(pArgs, 8, Py_BuildValue("i", 1));

  // 5. call the func
  pReturn = PyEval_CallObject(pFunc, pArgs);
  if (pReturn == NULL) {
    PyErr_Print();
    return "";
  }

  // 6. trans the result to C++
  std::string cutlass_op_name = "";
  std::string cutlass_op_def = "";
  objectsRepresentation = PyObject_Repr(PyTuple_GetItem(pReturn, 0));
  cutlass_op_name = PyUnicode_AsUTF8(objectsRepresentation);
  cutlass_op_name.erase(0, 1);
  cutlass_op_name.pop_back();
  std::cout << "cutlass_op_name is " << cutlass_op_name << std::endl;
  objectsRepresentation = PyObject_Repr(PyTuple_GetItem(pReturn, 1));
  cutlass_op_def = PyUnicode_AsUTF8(objectsRepresentation);
  cutlass_op_def.erase(0, 1);
  cutlass_op_def.pop_back();
  std::cout << "cutlass_op_def is " << cutlass_op_def << std::endl;

  // 7. finalize
  Py_DECREF(objectsRepresentation);
  Py_DECREF(pModule);
  Py_DECREF(pFunc);
  Py_DECREF(pArgs);
  Py_DECREF(pReturn);
  return cutlass_op_name;
}

std::shared_ptr<OpStrategy> StrategyForCutlassMatmul(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<Type> &out_type,
    const std::vector<std::vector<int>> &output_shapes,
    const Target &target) {
  framework::CINNCompute compute([=](lang::Args args, lang::RetValue *ret) {
    CHECK_EQ(args.size(), 1UL);
    CINNValuePack pack_args = args[0];
    CHECK_EQ(pack_args.size(), 2UL);
    CHECK(pack_args[0].is_string() && pack_args[1].is_string());
    CHECK_EQ(inputs.size(), 2);
    CHECK_EQ(output_shapes.size(), 1);
    CHECK_EQ(inputs[0]->shape.size(), 2);
    CHECK_EQ(inputs[1]->shape.size(), 2);
    std::string func_name = pack_args[0].operator std::string();
    std::string op_name = pack_args[1].operator std::string();
    std::string cutlass_kernel = GetCutlassKernel(op_name, inputs, out_type);

    // create call function.
    ir::Var kernel_ptr(cutlass_kernel + "_kernel", type_of<std::string>());
    ir::Var kernel_args(KERNEL_ARGS, type_of<void *>());

    std::vector<ir::Expr> host_args = {kernel_ptr, kernel_args};
    std::vector<ir::Argument> arguments = {
        ir::Argument(kernel_args, ir::Argument::IO::kOutput)};
    auto call_extern_api = ir::Call::Make(Void(),
                                          "cinn_call_cutlass",
                                          host_args,
                                          {},
                                          ir::CallType::Extern,
                                          ir::FunctionRef(),
                                          0);
    auto func =
        ir::_LoweredFunc_::Make(func_name, arguments, call_extern_api, {});

    VLOG(3) << func;
    *ret = CINNValuePack{{CINNValue(ir::Expr(func))}};
  });

  framework::CINNSchedule schedule(
      [=](lang::Args args, lang::RetValue *ret) {});

  auto strategy = std::make_shared<framework::OpStrategy>();
  strategy->AddImpl(compute, schedule, "strategy.cutlass_matmul.x86", 1);
  return strategy;
}

#ifdef CINN_WITH_CUDA
std::vector<ir::Expr> CutlassCallArgsForMatmul(
    const framework::NodeAttr &attrs,
    const std::vector<ir::Tensor> &inputs,
    const std::vector<std::vector<int>> &output_shapes) {
  CHECK_EQ(inputs.size(), 2);
  CHECK_EQ(output_shapes.size(), 1);
  CHECK_LE(inputs[0]->shape.size(), 4);
  CHECK_LE(inputs[1]->shape.size(), 4);

  const auto &attr_store = attrs.attr_store;
  bool trans_a = attr_store.count("trans_a")
                     ? absl::get<bool>(attr_store.at("trans_a"))
                     : false;
  bool trans_b = attr_store.count("trans_b")
                     ? absl::get<bool>(attr_store.at("trans_b"))
                     : false;
  bool trans_out = attr_store.count("trans_out")
                       ? absl::get<bool>(attr_store.at("trans_out"))
                       : false;
  float alpha = attr_store.count("alpha")
                    ? absl::get<float>(attr_store.at("alpha"))
                    : 1.0f;
  float beta =
      attr_store.count("beta") ? absl::get<float>(attr_store.at("beta")) : 0.0f;

  int x_num_col_dims = attr_store.count("x_num_col_dims")
                           ? absl::get<int>(attr_store.at("x_num_col_dims"))
                           : 0;
  int y_num_col_dims = attr_store.count("y_num_col_dims")
                           ? absl::get<int>(attr_store.at("y_num_col_dims"))
                           : 0;
  bool is_infer = attr_store.count("is_infer")
                      ? absl::get<bool>(attr_store.at("is_infer"))
                      : false;
  CHECK((x_num_col_dims == 0 && y_num_col_dims == 0) ||
        (x_num_col_dims > 0 && y_num_col_dims > 0));

  std::vector<ir::Expr> a_shape, b_shape;
  if (x_num_col_dims == 0 && y_num_col_dims == 0) {
    int a_rank = inputs[0]->shape.size();
    int b_rank = inputs[1]->shape.size();

    if (a_rank == 1) {
      a_shape.resize(4, ir::Expr(1));

      if (trans_a) {
        a_shape[2] = inputs[0]->shape[0];
      } else {
        a_shape[3] = inputs[0]->shape[0];
      }
    } else {
      a_shape = inputs[0]->shape;
      int insert_1_to_a = 4 - a_shape.size();
      for (int idx = 0; idx < insert_1_to_a; ++idx) {
        a_shape.insert(a_shape.begin(), ir::Expr(1));
      }
    }

    if (b_rank == 1) {
      b_shape.resize(4, ir::Expr(1));

      if (trans_b) {
        b_shape[3] = inputs[1]->shape[0];
      } else {
        b_shape[2] = inputs[1]->shape[0];
      }
    } else {
      b_shape = inputs[1]->shape;
      int insert_1_to_b = 4 - b_shape.size();
      for (int idx = 0; idx < insert_1_to_b; ++idx) {
        b_shape.insert(b_shape.begin(), ir::Expr(1));
      }
    }
  } else if (x_num_col_dims > 0 && y_num_col_dims > 0) {
    // input a shape.
    a_shape = {Expr(1), Expr(1)};
    int a_height = 1;
    int a_width = 1;
    for (int idx = 0; idx < x_num_col_dims; ++idx) {
      a_height *= inputs[0]->shape[idx].as_int32();
    }
    for (int idx = x_num_col_dims; idx < inputs[0]->shape.size(); ++idx) {
      a_width *= inputs[0]->shape[idx].as_int32();
    }
    a_shape.emplace_back(a_height);
    a_shape.emplace_back(a_width);

    // input b shape.
    b_shape = {Expr(1), Expr(1)};
    int b_height = 1;
    int b_width = 1;
    for (int idx = 0; idx < y_num_col_dims; ++idx) {
      b_height *= inputs[1]->shape[idx].as_int32();
    }
    for (int idx = y_num_col_dims; idx < inputs[1]->shape.size(); ++idx) {
      b_width *= inputs[1]->shape[idx].as_int32();
    }
    b_shape.emplace_back(b_height);
    b_shape.emplace_back(b_width);

    if (is_infer) {
      CHECK_EQ(a_width, b_width)
          << "The K dimension of mul shold be equal! Please check.";
      trans_b = true;
    } else {
      CHECK_EQ(a_width, b_height)
          << "The K dimension of mul shold be equal! Please check.";
    }
  } else {
    LOG(FATAL) << "Unkown Matmul Setting!";
  }

  CHECK_EQ(a_shape.size(), 4);
  CHECK_EQ(b_shape.size(), 4);
  // func args
  std::vector<ir::Expr> args = {ir::Expr(trans_a),
                                ir::Expr(trans_b),
                                ir::Expr(trans_out),
                                ir::Expr(alpha),
                                ir::Expr(beta)};
  args.insert(args.end(), a_shape.begin(), a_shape.end());
  args.insert(args.end(), b_shape.begin(), b_shape.end());
  return args;
}
#endif

bool RegisteryCutlassCallArgsFunc() {
#ifdef CINN_WITH_CUDA
  CutlassCallArgsFuncRegistry::Global().Register(
      "cutlass_matmul", common::DefaultNVGPUTarget(), CutlassCallArgsForMatmul);
#endif

  return true;
}

static bool registry_cutlass_call_list_func = RegisteryCutlassCallArgsFunc();
}  // namespace op
}  // namespace hlir
}  // namespace cinn

CINN_REGISTER_HELPER(cutlass_ops) {
  CINN_REGISTER_OP(cutlass_matmul)
      .describe("This operator implements the call of cutlass kernels!")
      .set_attr<cinn::hlir::framework::StrategyFunction>(
          "CINNStrategy", cinn::hlir::op::StrategyForCutlassMatmul)
      .set_attr<cinn::hlir::framework::OpPatternKind>(
          "OpPattern", cinn::hlir::framework::OpPatternKind::kNonFusible);

  return true;
}
