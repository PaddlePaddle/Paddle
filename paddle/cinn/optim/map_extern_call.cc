// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/optim/map_extern_call.h"

#include "paddle/cinn/cinn.h"
#include "paddle/cinn/hlir/op/op_util.h"
#include "paddle/cinn/ir/ir_mutator.h"
#include "paddle/cinn/runtime/cpu/host_intrinsics.h"

namespace cinn {
namespace optim {

static const std::set<std::string> kExternFp32CallsGPU{
    {"exp",   "erf",   "sigmoid", "sqrt",     "log",   "log2",      "log10",
     "floor", "ceil",  "round",   "trunc",    "cos",   "cosh",      "tan",
     "sin",   "sinh",  "acos",    "acosh",    "asin",  "asinh",     "atan",
     "atanh", "isnan", "tanh",    "isfinite", "isinf", "remainder", "rsqrt",
     "cbrt",  "abs",   "pow",     "mod"}};

static const std::set<std::string> kExternInt32CallsGPU{{"left_shift",
                                                         "right_shift",
                                                         "bitwise_or",
                                                         "bitwise_and",
                                                         "bitwise_xor",
                                                         "bitwise_not",
                                                         "pow",
                                                         "logical_right_shift",
                                                         "clz",
                                                         "popc",
                                                         "mod"}};

static const std::set<std::string> kExternFp32CallsCPU = {
    "erf", "acos", "acosh", "asin", "asinh", "atan", "atanh", "remainder"};

void MapExternCall(Expr *e, Target target) {
  struct Mutator : ir::IRMutator<Expr *> {
    Target target;

    explicit Mutator(Target target) : target(target) {}

    void operator()(Expr *e) { ir::IRMutator<>::Visit(e, e); }

    void Visit(const ir::Call *op, Expr *expr) override {
      auto *node = expr->As<ir::Call>();
      CHECK(node);
      OptimizeConstantPow(node);
      if (target.arch == Target::Arch::NVGPU) {
        DealWithNvGpuintrinsics(node, expr);
      } else {
        DealWithCpuintrinsics(node, expr);
      }
    }

    void DealWithCpuintrinsics(ir::Call *node, Expr *expr) {
      if (kExternFp32CallsCPU.count(node->name)) {
        CHECK_GE(node->read_args.size(), 1UL);
        CHECK(node->read_args.front().type().is_float())
            << "CPU extern call instrinsices only support float now! Please "
               "check.";
        if (node->read_args.front().type().is_float(32)) {
          auto out_type = node->type();
          *expr = lang::CallExtern(node->name + "f", node->read_args);
        }
      }
    }

    void DealWithNvGpuintrinsics(ir::Call *node, Expr *expr) {
      auto arg_size = node->read_args.size();
      if (arg_size == 0UL) {
        // some node like __syncthreads hasn't arguments
        return;
      }
      const auto &dtype = node->read_args.front().type();
      const auto &name = node->name;

      bool node_in_extern_fp32 = kExternFp32CallsGPU.count(name);
      bool node_in_extern_int32 = kExternInt32CallsGPU.count(name);
      if (!node_in_extern_fp32 && !node_in_extern_int32) {
        return;
      }

      std::string extern_func =
          hlir::GetExternFuncName(common::DefaultNVGPUTarget(), dtype, name);
      *expr = lang::CallExtern(extern_func, node->read_args, node->attrs);
    }

    // Replace pow(x, 0.5) to sqrt(x) and pow(x, -0.5) to rsqrt(x), which
    // can speed up a lot.
    //
    // Reference:
    // https://en.wikipedia.org/wiki/Fast_inverse_square_root
    void OptimizeConstantPow(ir::Call *node) {
      if (node->name == "pow" && node->read_args.size() >= 2 &&
          node->read_args[1].is_constant()) {
        float pow_constant = node->read_args[1].get_constant();
        if (pow_constant == 0.5) {
          node->name = "sqrt";
          node->read_args.erase(node->read_args.begin() + 1);
        } else if (pow_constant == -0.5) {
          node->name = "rsqrt";
          node->read_args.erase(node->read_args.begin() + 1);
        }
      }
    }
  };

  Mutator m(target);
  m(e);
}

}  // namespace optim
}  // namespace cinn
