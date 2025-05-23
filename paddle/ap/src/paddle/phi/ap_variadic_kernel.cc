// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/ap/include/kernel_dispatch/ap_variadic_kernel.h"

#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include "glog/logging.h"
#include "paddle/common/enforce.h"

#include "paddle/ap/include/axpr/anf_expr_util.h"
#include "paddle/ap/include/fs/builtin_functions.h"
#include "paddle/ap/include/kernel_dispatch/builtin_frame_util.h"
#include "paddle/ap/include/paddle/phi/kernel_define_helper.h"
#include "paddle/ap/include/paddle/phi/kernel_dispatch_helper.h"
#include "paddle/ap/include/rt_module/naive_module_maker.h"

namespace ap {

using MakeCoreExprT = adt::Result<ap::axpr::Lambda<ap::axpr::CoreExpr>> (*)(
    const std::string& json_str);

adt::Result<ap::axpr::Lambda<ap::axpr::CoreExpr>> ConvertToCoreExpr(
    const std::string& json_str) {
  const auto& anf_expr = ap::axpr::MakeAnfExprFromJsonString(json_str);
  ADT_RETURN_IF_ERR(anf_expr);
  const auto& core_expr =
      ap::axpr::ConvertAnfExprToCoreExpr(anf_expr.GetOkValue());
  if (!core_expr.Has<ap::axpr::Atomic<ap::axpr::CoreExpr>>()) {
    return adt::errors::TypeError{
        std::string() + "json_str can not be converted to atomic AnfExpr."};
  }
  const auto& atomic = core_expr.Get<ap::axpr::Atomic<ap::axpr::CoreExpr>>();
  if (!atomic.Has<ap::axpr::Lambda<ap::axpr::CoreExpr>>()) {
    return adt::errors::TypeError{
        std::string() + "json_str can not be converted to lambda AnfExpr."};
  }
  return atomic.Get<ap::axpr::Lambda<ap::axpr::CoreExpr>>();
}

template <MakeCoreExprT MakeCoreExpr>
adt::Result<ap::axpr::Lambda<ap::axpr::CoreExpr>> CacheCoreExpr(
    const std::string& json_str) {
  static std::unordered_map<std::string,
                            adt::Result<ap::axpr::Lambda<ap::axpr::CoreExpr>>>
      json_str2cache;
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);
  auto iter = json_str2cache.find(json_str);
  if (iter == json_str2cache.end()) {
    const auto& core_expr = MakeCoreExpr(json_str);
    iter = json_str2cache.emplace(json_str, core_expr).first;
  }
  return iter->second;
}

constexpr MakeCoreExprT MakeOrGetCoreExpr = &CacheCoreExpr<&ConvertToCoreExpr>;

namespace kernel_dispatch {

using FuncName2ArgTypes =
    std::unordered_map<std::string, adt::List<code_module::ArgType>>;
FuncName2ArgTypes MakeFuncName2ArgTypes(const code_module::CodeModule& m) {
  auto GetArgTypes = [&](const auto& declare) { return declare->arg_types; };
  FuncName2ArgTypes ret;
  for (const auto& declare : *m->func_declares) {
    ret[declare->func_id] = GetArgTypes(declare);
  }
  return ret;
}

using MakeRtModuleT = adt::Result<kernel_dispatch::RtModule> (*)(
    const std::string& code_module_lambda);

template <MakeRtModuleT MakeRtModule>
adt::Result<kernel_dispatch::RtModule> CacheRtModule(
    const std::string& code_module_lambda) {
  using Definer2RtModule =
      std::unordered_map<std::string, adt::Result<kernel_dispatch::RtModule>>;
  static Definer2RtModule definer2rt_module;
  static std::mutex mutex;
  std::unique_lock<std::mutex> lock(mutex);
  auto iter = definer2rt_module.find(code_module_lambda);
  if (iter == definer2rt_module.end()) {
    const auto& rt_module = MakeRtModule(code_module_lambda);
    iter = definer2rt_module.emplace(code_module_lambda, rt_module).first;
  }
  return iter->second;
}

adt::Result<kernel_dispatch::RtModule> MakeRtModule(
    const std::string& code_module_lambda) {
  ADT_LET_CONST_REF(code_module_core_expr,
                    MakeOrGetCoreExpr(code_module_lambda));
  phi::KernelDefineHelper helper{};
  ADT_LET_CONST_REF(code_module,
                    helper.InterpretKernelDefineLambda(code_module_core_expr));
  using RetT = adt::Result<kernel_dispatch::RtModule>;
  return code_module->source_code.Match(
      [&](const ap::code_module::Project&) -> RetT {
        const char* ap_workspace_dir = std::getenv("AP_WORKSPACE_DIR");
        ADT_CHECK(ap_workspace_dir != nullptr) << adt::errors::TypeError{
            std::string() + "AP_WORKSPACE_DIR not set"};
        auto hash_value_str =
            std::to_string(std::hash<std::string>()(code_module_lambda));
        std::string workspace_dir =
            std::string(ap_workspace_dir) + "/" + hash_value_str;
        ap::rt_module::NaiveModuleMaker maker(workspace_dir);
        auto Serialize = [&](const auto&) -> const std::string& {
          return code_module_lambda;
        };
        ADT_LET_CONST_REF(rt_module, maker.Make(code_module, Serialize));
        return rt_module;
      },
      [&](const ap::code_module::Package&) -> RetT {
        const char* ap_workspace_dir = std::getenv("AP_PACKAGE_DIR");
        ap::rt_module::NaiveModuleMaker maker(ap_workspace_dir);
        auto Serialize = [&](const auto&) -> const std::string& {
          return code_module_lambda;
        };
        ADT_LET_CONST_REF(rt_module, maker.Make(code_module, Serialize));
        return rt_module;
      });
}

constexpr MakeRtModuleT MakeOrGetRtModule = &CacheRtModule<&MakeRtModule>;

adt::List<Val> MakeTensorDims(const phi::DenseTensor& tensor) {
  adt::List<Val> ret;
  ret->reserve(tensor.dims().size());
  for (int i = 0; i < tensor.dims().size(); ++i) {
    ret->emplace_back(Val{tensor.dims().at(i)});
  }
  return ret;
}

adt::Result<adt::List<ap::axpr::SerializableValue>> GetIndexesSlices(
    const ap::axpr::AttrMap<ap::axpr::SerializableValue>&
        kernel_dispatch_const_data,
    const std::string& attr_name) {
  ADT_LET_CONST_REF(
      val,
      kernel_dispatch_const_data
          ->TryGet<adt::List<ap::axpr::SerializableValue>>(attr_name));
  return val;
}

template <typename DoEachIdxT, typename DoEachRangeT>
adt::Result<adt::Ok> VisitTensorIdxOrRange(
    const adt::List<ap::axpr::SerializableValue>& list,
    const DoEachIdxT& DoEachIdx,
    const DoEachRangeT& DoEachRange) {
  using Ok = adt::Result<adt::Ok>;
  for (size_t i = 0; i < list->size(); ++i) {
    const auto& elt = list->at(i);
    ADT_RETURN_IF_ERR(elt.Match(
        [&](int64_t idx) -> Ok {
          ADT_RETURN_IF_ERR(DoEachIdx(idx));
          return adt::Ok{};
        },
        [&](const adt::List<ap::axpr::SerializableValue>& range_val) -> Ok {
          ADT_CHECK(range_val->size() == 2);
          ADT_LET_CONST_REF(start, range_val->at(0).TryGet<int64_t>());
          ADT_LET_CONST_REF(end, range_val->at(1).TryGet<int64_t>());
          ADT_RETURN_IF_ERR(DoEachRange(start, end));
          return adt::Ok{};
        },
        [&](const auto&) -> Ok {
          return adt::errors::TypeError{"only index or index pair supported."};
        }));
  }
  return adt::Ok{};
}

adt::Result<adt::List<Val>> MakeConstTensors(
    const std::vector<const phi::DenseTensor*>& xs,
    const ap::axpr::AttrMap<ap::axpr::SerializableValue>&
        kernel_dispatch_const_data) {
  ADT_LET_CONST_REF(
      indexes_slices,
      GetIndexesSlices(kernel_dispatch_const_data,
                       "__builtin_ap_kernel_input_indexes_slices"));
  adt::List<Val> ret;
  ret->reserve(xs.size());
  using Ok = adt::Result<adt::Ok>;
  auto CollectTensor = [&](adt::List<Val>* list,
                           const phi::DenseTensor* x) -> Ok {
    ConstTensorData tensor_data{x};
    adt::List<Val> dims{MakeTensorDims(*x)};
    ConstTensor<Val> const_tensor{tensor_data, dims};
    axpr::BuiltinClassInstance<Val> instance{GetConstTensorClass<Val>(),
                                             const_tensor};
    (*list)->emplace_back(instance);
    return adt::Ok{};
  };
  auto DoEachIdx = [&](std::size_t i) -> Ok {
    ADT_CHECK(i < xs.size());
    const auto* x = xs.at(i);
    ADT_RETURN_IF_ERR(CollectTensor(&ret, x));
    return adt::Ok{};
  };
  auto DoEachRange = [&](std::size_t start, std::size_t end) -> Ok {
    ADT_CHECK(start <= end);
    adt::List<Val> tensor_list;
    tensor_list->reserve(end - start);
    for (size_t i = start; i < end; ++i) {
      ADT_CHECK(i < xs.size());
      const auto* x = xs.at(i);
      ADT_RETURN_IF_ERR(CollectTensor(&tensor_list, x));
    }
    ret->emplace_back(tensor_list);
    return adt::Ok{};
  };
  ADT_RETURN_IF_ERR(
      VisitTensorIdxOrRange(indexes_slices, DoEachIdx, DoEachRange));
  return ret;
}

adt::Result<adt::List<Val>> MakeMutableTensors(
    const std::vector<phi::DenseTensor*>& xs,
    const ap::axpr::AttrMap<ap::axpr::SerializableValue>&
        kernel_dispatch_const_data) {
  ADT_LET_CONST_REF(
      indexes_slices,
      GetIndexesSlices(kernel_dispatch_const_data,
                       "__builtin_ap_kernel_output_indexes_slices"));
  adt::List<Val> ret;
  ret->reserve(xs.size());

  using Ok = adt::Result<adt::Ok>;
  auto CollectTensor = [&](adt::List<Val>* list, phi::DenseTensor* x) -> Ok {
    MutableTensorData tensor_data{x};
    adt::List<Val> dims{MakeTensorDims(*x)};
    MutableTensor<Val> mutable_tensor{tensor_data, dims};
    axpr::BuiltinClassInstance<Val> instance{GetMutableTensorClass<Val>(),
                                             mutable_tensor};
    (*list)->emplace_back(instance);
    return adt::Ok{};
  };
  auto DoEachIdx = [&](std::size_t i) -> Ok {
    ADT_CHECK(i < xs.size());
    auto* x = xs.at(i);
    ADT_RETURN_IF_ERR(CollectTensor(&ret, x));
    return adt::Ok{};
  };
  auto DoEachRange = [&](std::size_t start, std::size_t end) -> Ok {
    ADT_CHECK(start <= end);
    adt::List<Val> tensor_list;
    tensor_list->reserve(end - start);
    for (size_t i = start; i < end; ++i) {
      ADT_CHECK(i < xs.size());
      auto* x = xs.at(i);
      ADT_RETURN_IF_ERR(CollectTensor(&tensor_list, x));
    }
    ret->emplace_back(tensor_list);
    return adt::Ok{};
  };
  ADT_RETURN_IF_ERR(
      VisitTensorIdxOrRange(indexes_slices, DoEachIdx, DoEachRange));
  return ret;
}

adt::Result<adt::Ok> ApVariadicKernel(
    const DeviceCtx& device_ctx,
    const std::vector<const phi::DenseTensor*>& xs,
    int num_outputs,
    const std::string& code_module_lambda,
    const std::string& infer_meta_lambda,
    const std::string& kernel_dispatch_lambda,
    const std::string& kernel_dispatch_const_data_lambda,
    std::vector<phi::DenseTensor*> outs) {
  phi::KernelDispatchHelper helper{};
  ADT_LET_CONST_REF(ctx_maker_lambda,
                    MakeOrGetCoreExpr(kernel_dispatch_const_data_lambda));
  ADT_LET_CONST_REF(ctx_maker_ret, helper.InterpretCtxMaker(ctx_maker_lambda));
  ADT_LET_CONST_REF(
      kernel_dispatch_const_data,
      ctx_maker_ret.TryGet<ap::axpr::AttrMap<ap::axpr::SerializableValue>>());
  ADT_LET_CONST_REF(rt_module,
                    kernel_dispatch::MakeOrGetRtModule(code_module_lambda));
  ADT_LET_CONST_REF(inputs, MakeConstTensors(xs, kernel_dispatch_const_data));
  ADT_LET_CONST_REF(outputs,
                    MakeMutableTensors(outs, kernel_dispatch_const_data));
  DispatchRawCtx<Val> raw_ctx{device_ctx, inputs, outputs, rt_module};
  DispatchCtx<Val> dispatch_ctx{raw_ctx, kernel_dispatch_const_data};
  ADT_LET_CONST_REF(lambda, MakeOrGetCoreExpr(kernel_dispatch_lambda));
  ADT_RETURN_IF_ERR(helper.InterpretKernelDispatcher(lambda, dispatch_ctx));
  return adt::Ok{};
}

}  // namespace kernel_dispatch

}  // namespace ap
