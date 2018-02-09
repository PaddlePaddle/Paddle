/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/pybind/protobuf.h"

#include <mutex>  // for call_once
#include <unordered_map>
#include "paddle/framework/backward.h"
#include "paddle/framework/executor.h"
#include "paddle/framework/feed_fetch_method.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/init.h"
#include "paddle/framework/lod_rank_table.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/lod_tensor_array.h"
#include "paddle/framework/prune.h"
#include "paddle/framework/selected_rows.h"
#include "paddle/operators/cond_op.h"
#include "paddle/operators/net_op.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/place.h"
#include "paddle/platform/profiler.h"
#include "paddle/pybind/const_value.h"
#include "paddle/pybind/exception.h"
#include "paddle/pybind/pybind.h"
#include "paddle/pybind/tensor_py.h"
#include "paddle/string/to_string.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/operators/nccl/nccl_gpu_common.h"
#include "paddle/platform/cuda_profiler.h"
#include "paddle/platform/gpu_info.h"
#endif

// disable auto conversion to list in Python
PYBIND11_MAKE_OPAQUE(paddle::framework::LoDTensorArray);

namespace paddle {
namespace pybind {
static size_t UniqueIntegerGenerator(const std::string &prefix) {
  static std::unordered_map<std::string, std::atomic<size_t>> generators;
  return generators[prefix].fetch_add(1);
}

bool IsCompiledWithCUDA() {
#ifndef PADDLE_WITH_CUDA
  return false;
#else
  return true;
#endif
}

PYBIND11_PLUGIN(core) {
  py::module m("core", "C++ core of PaddlePaddle");

  // using framework in this function. Since it is inside a function, it will
  // not cause namespace pollution.
  using namespace paddle::framework;  // NOLINT

  BindException(m);

  py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
      .def_buffer(
          [](Tensor &self) -> py::buffer_info { return CastToPyBuffer(self); })
      .def("get_dims",
           [](const Tensor &self) { return vectorize(self.dims()); })
      .def("set_dims",
           [](Tensor &self, const std::vector<int64_t> &dim) {
             self.Resize(make_ddim(dim));
           })
      .def("set_layout",
           [](Tensor &self, const std::string &layout) {
             self.set_layout(StringToDataLayout(layout));
           })
      .def("alloc_float",
           [](Tensor &self, paddle::platform::CUDAPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("alloc_float",
           [](Tensor &self, paddle::platform::CPUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("alloc_int",
           [](Tensor &self, paddle::platform::CPUPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("alloc_int",
           [](Tensor &self, paddle::platform::CUDAPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("set", PyCPUTensorSetFromArray<float>)
      .def("set", PyCPUTensorSetFromArray<int>)
      .def("set", PyCPUTensorSetFromArray<double>)
      .def("set", PyCPUTensorSetFromArray<int64_t>)
      .def("set", PyCPUTensorSetFromArray<bool>)
#ifdef PADDLE_WITH_CUDA
      .def("set", PyCUDATensorSetFromArray<float>)
      .def("set", PyCUDATensorSetFromArray<int>)
      .def("set", PyCUDATensorSetFromArray<double>)
      .def("set", PyCUDATensorSetFromArray<int64_t>)
      .def("set", PyCUDATensorSetFromArray<bool>)
#endif
      .def("shape", [](Tensor &self) { return vectorize(self.dims()); })
      .def("set_float_element", TensorSetElement<float>)
      .def("get_float_element", TensorGetElement<float>)
      .def("set_double_element", TensorSetElement<double>)
      .def("get_double_element", TensorGetElement<double>)
      .def("dtype", [](Tensor &self) { return ToDataType(self.type()); });

  py::class_<LoDTensor, Tensor>(m, "LoDTensor")
      .def_buffer(
          [](Tensor &self) -> py::buffer_info { return CastToPyBuffer(self); })
      .def(
          "__init__",
          [](LoDTensor &instance, const std::vector<std::vector<size_t>> &lod) {
            LoD new_lod;
            new_lod.reserve(lod.size());
            std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
            new (&instance) LoDTensor(new_lod);
          })
      .def("__init__", [](LoDTensor &instance) { new (&instance) LoDTensor(); })
      .def("set_lod",
           [](LoDTensor &self, const std::vector<std::vector<size_t>> &lod) {
             LoD new_lod;
             new_lod.reserve(lod.size());
             std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
             self.set_lod(new_lod);
           })
      .def("lod", [](LoDTensor &self) -> std::vector<std::vector<size_t>> {
        auto lod = self.lod();
        std::vector<std::vector<size_t>> new_lod;
        new_lod.reserve(lod.size());
        std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
        return new_lod;
      });

  py::class_<SelectedRows>(m, "SelectedRows")
      .def("__init__",
           [](SelectedRows &instance) { new (&instance) SelectedRows(); })
      .def("__init__",
           [](SelectedRows &instance, const std::vector<int64_t> rows,
              const int64_t &height) {
             new (&instance) SelectedRows(rows, height);
           })
      .def("get_tensor",
           [](SelectedRows &self) { return self.mutable_value(); },
           py::return_value_policy::reference)
      .def("set_height", &SelectedRows::set_height)
      .def("height", &SelectedRows::height)
      .def("set_rows",
           [](SelectedRows &self, std::vector<int64_t> rows) {
#ifndef PADDLE_WITH_CUDA
             self.set_rows(rows);
#else
        Vector<int64_t> new_rows(rows);
        self.set_rows(new_rows);
#endif
           })
      .def("rows", [](SelectedRows &self) {
#ifndef PADDLE_WITH_CUDA
        return self.rows();
#else
         auto rows = self.rows();
         std::vector<int64_t> new_rows;
         new_rows.reserve(rows.size());
         std::copy(rows.begin(), rows.end(), std::back_inserter(new_rows));
         return new_rows;
#endif
      });

  py::class_<Variable>(m, "Variable", R"DOC(Variable Class.

All parameter, weight, gradient are variables in Paddle.
)DOC")
      .def("is_int", [](const Variable &var) { return var.IsType<int>(); })
      .def("set_int",
           [](Variable &var, int val) -> void { *var.GetMutable<int>() = val; })
      .def("get_int", [](const Variable &var) -> int { return var.Get<int>(); })
      .def("is_float", [](const Variable &var) { return var.IsType<float>(); })
      .def("set_float",
           [](Variable &var, float val) -> void {
             *var.GetMutable<float>() = val;
           })
      .def("get_float",
           [](const Variable &var) -> float { return var.Get<float>(); })
      .def("get_tensor",
           [](Variable &self) -> LoDTensor * {
             return self.GetMutable<LoDTensor>();
           },
           py::return_value_policy::reference)
      .def("get_lod_rank_table",
           [](Variable &self) { return self.GetMutable<LoDRankTable>(); },
           py::return_value_policy::reference)
      .def("get_selected_rows",
           [](Variable &self) -> SelectedRows * {
             return self.GetMutable<SelectedRows>();
           },
           py::return_value_policy::reference)
      .def("get_lod_tensor_array",
           [](Variable &self) { return self.GetMutable<LoDTensorArray>(); },
           py::return_value_policy::reference)
#ifdef PADDLE_WITH_CUDA
      .def("get_communicator",
           [](Variable &self) -> platform::Communicator * {
             return self.GetMutable<platform::Communicator>();
           },
           py::return_value_policy::reference)
#endif
      .def("get_net",
           [](Variable &self) -> operators::NetOp * {
             return self.GetMutable<operators::NetOp>();
           },
           py::return_value_policy::reference);

  py::class_<Scope>(m, "Scope", "")
      .def("var",
           [](Scope &self, const std::string &name) -> Variable * {
             return self.Var(name);
           },
           py::return_value_policy::reference)
      .def("find_var", &Scope::FindVar, py::return_value_policy::reference)
      .def(py::init<>())
      .def("new_scope", [](Scope &self) -> Scope * { return &self.NewScope(); },
           py::return_value_policy::reference)
      .def("drop_kids", &Scope::DropKids);

  //! @note: Be careful! PyBind will return std::string as an unicode, not
  //! Python str. If you want a str object, you should cast them in Python.
  m.def("get_all_op_protos", []() -> std::vector<py::bytes> {
    std::vector<py::bytes> ret_values;
    for (auto &iter : OpInfoMap::Instance().map()) {
      auto &info = iter.second;
      if (info.HasOpProtoAndChecker()) {
        std::string str;
        PADDLE_ENFORCE(
            info.Proto().SerializeToString(&str),
            "Serialize OpProto Error. This could be a bug of Paddle.");
        ret_values.emplace_back(str);
      }
    }
    return ret_values;
  });
  m.def(
      "get_grad_op_desc", [](const OpDesc &op_desc,
                             const std::unordered_set<std::string> &no_grad_set,
                             const std::vector<BlockDesc *> &grad_sub_block) {
        std::unordered_map<std::string, std::string> grad_to_var;
        std::vector<std::unique_ptr<OpDesc>> grad_op_descs =
            framework::OpInfoMap::Instance()
                .Get(op_desc.Type())
                .GradOpMaker()(op_desc, no_grad_set, &grad_to_var,
                               grad_sub_block);
        std::vector<OpDesc *> grad_op_desc_ptrs(grad_op_descs.size());
        std::transform(grad_op_descs.begin(), grad_op_descs.end(),
                       grad_op_desc_ptrs.begin(),
                       [](std::unique_ptr<OpDesc> &p) { return p.release(); });
        return std::make_pair(grad_op_desc_ptrs, grad_to_var);
      });
  m.def("prune", [](const ProgramDesc &origin,
                    const std::vector<std::array<size_t, 2>> &targets) {
    ProgramDesc prog_with_targets(origin);
    for (const auto &t : targets) {
      prog_with_targets.MutableBlock(t[0])->Op(t[1])->MarkAsTarget();
    }
    proto::ProgramDesc pruned_desc;
    Prune(*prog_with_targets.Proto(), &pruned_desc);
    return new ProgramDesc(pruned_desc);
  });
  m.def("inference_optimize", [](ProgramDesc &origin) {
    proto::ProgramDesc pruned_desc;
    InferenceOptimize(*(origin.Proto()), &pruned_desc);
    return new ProgramDesc(pruned_desc);
  });
  m.def("empty_var_name", []() { return framework::kEmptyVarName; });
  m.def("grad_var_suffix", []() { return framework::kGradVarSuffix; });
  m.def_submodule(
       "var_names",
       "The module will return special predefined variable name in Paddle")
      .def("empty", []() { return kEmptyVarName; })
      .def("temp", []() { return kTempVarName; });
  // clang-format off
  py::class_<paddle::platform::DeviceContext>(m, "DeviceContext")
      .def_static("create",
                  [](paddle::platform::CPUPlace& place)
                      -> paddle::platform::DeviceContext* {
                    return new paddle::platform::CPUDeviceContext();
                  })
      .def_static("create",
                  [](paddle::platform::CUDAPlace& place)
                      -> paddle::platform::DeviceContext* {
#ifndef PADDLE_WITH_CUDA
                    PADDLE_THROW("CUDAPlace is not supported in CPU device.");
#else
                    return new paddle::platform::CUDADeviceContext(place);
#endif
                  });
// clang-format on

#ifdef PADDLE_WITH_CUDA
  py::class_<platform::Communicator>(m, "Communicator").def(py::init<>());
#endif
  py::class_<platform::CUDAPlace>(m, "CUDAPlace")
      .def(py::init<int>())
      .def("__str__", string::to_string<const platform::CUDAPlace &>);

  py::class_<paddle::platform::CPUPlace>(m, "CPUPlace")
      .def(py::init<>())
      .def("__str__", string::to_string<const platform::CPUPlace &>);

  py::class_<platform::Place>(m, "Place")
      .def(py::init<>())
      .def("set_place",
           [](platform::Place &self, const platform::CPUPlace &cpu_place) {
             self = cpu_place;
           })
      .def("set_place",
           [](platform::Place &self, const platform::CUDAPlace &gpu_place) {
             self = gpu_place;
           });

  py::class_<OperatorBase>(m, "Operator")
      .def_static("create",
                  [](py::bytes protobin) {
                    proto::OpDesc desc;
                    PADDLE_ENFORCE(desc.ParsePartialFromString(protobin),
                                   "Cannot parse user input to OpDesc");
                    PADDLE_ENFORCE(desc.IsInitialized(),
                                   "User OpDesc is not initialized, reason %s",
                                   desc.InitializationErrorString());
                    return OpRegistry::CreateOp(desc);
                  })
      .def("backward",
           [](const OperatorBase &forwardOp,
              const std::unordered_set<std::string> &no_grad_vars) {
             return Backward(forwardOp, no_grad_vars).release();
           })
      .def("run",
           [](OperatorBase &self, const Scope &scope,
              const platform::CPUPlace &place) { self.Run(scope, place); })
      .def("run",
           [](OperatorBase &self, const Scope &scope,
              const platform::CUDAPlace &place) { self.Run(scope, place); })
      .def("type",
           [](const OperatorBase &op) -> std::string { return op.Type(); })
      .def("outputs",
           [](const OperatorBase &op)
               -> std::map<std::string, std::vector<std::string>> {
                 return op.Outputs();
               })
      .def("output_vars",
           [](const OperatorBase &op) { return op.OutputVars(true); })
      .def("inputs", [](const OperatorBase &op) { return op.Inputs(); })
      .def("input_vars", [](const OperatorBase &op) { return op.InputVars(); })
      .def("__str__", &OperatorBase::DebugString)
      .def("no_intermediate_outputs",
           [](const OperatorBase &op) { return op.OutputVars(false); })
      .def("support_gpu", &OperatorBase::SupportGPU);

  py::class_<operators::NetOp, OperatorBase>(m, "Net")
      .def_static("create",
                  []() -> operators::NetOp * {
                    auto *retv = new operators::NetOp;
                    retv->SetType("plain_net");
                    return retv;
                  })
      .def("append_op", [](operators::NetOp &self,
                           const OperatorBase &op) { self.AppendOp(op); })
      .def("complete_add_op", &operators::NetOp::CompleteAddOp)
      .def("complete_add_op", [](std::shared_ptr<operators::NetOp> &self) {
        self->CompleteAddOp();
      });

  // cond_op
  py::class_<operators::CondOp, OperatorBase>(m, "CondOp")
      .def_static("create",
                  [](py::bytes protobin) -> operators::CondOp * {
                    proto::OpDesc desc;
                    PADDLE_ENFORCE(desc.ParsePartialFromString(protobin),
                                   "Cannot parse user input to OpDesc");
                    PADDLE_ENFORCE(desc.IsInitialized(),
                                   "User OpDesc is not initialized, reason %s",
                                   desc.InitializationErrorString());
                    auto cond_op = OpRegistry::CreateOp(desc);
                    return static_cast<operators::CondOp *>(cond_op.release());
                  })
      .def("set_truenet",
           [](operators::CondOp &self, const operators::NetOp &net) -> void {
             self.set_truenet(net.Clone());
           })
      .def("set_falsenet",
           [](operators::CondOp &self, const operators::NetOp &net) -> void {
             self.set_falsenet(net.Clone());
           });

  py::class_<framework::Executor>(m, "Executor")
      .def(py::init<const platform::Place &>())
      .def("run",
           (void (Executor::*)(const ProgramDesc &, Scope *, int, bool, bool)) &
               Executor::Run);

  m.def("unique_integer", UniqueIntegerGenerator);
  m.def("init_gflags", framework::InitGflags);
  m.def("init_glog", framework::InitGLOG);
  m.def("init_devices", &framework::InitDevices);

  m.def("is_compiled_with_cuda", IsCompiledWithCUDA);

  m.def("set_feed_variable", framework::SetFeedVariable);
  m.def("get_fetch_variable", framework::GetFetchVariable);

  BindProgramDesc(m);
  BindBlockDesc(m);
  BindVarDsec(m);
  BindOpDesc(m);
  BindConstValue(m);

  py::class_<framework::LoDRankTable>(m, "LodRankTable")
      .def("items", [](framework::LoDRankTable &table) {
        std::vector<std::pair<size_t, size_t>> res;
        for (auto &item : table.items()) {
          res.push_back({item.index, item.length});
        }
        return res;
      });

  py::class_<LoDTensorArray>(m, "LoDTensorArray")
      .def("__getitem__",
           [](LoDTensorArray &self, size_t i) { return &self.at(i); },
           py::return_value_policy::reference)
      .def("__len__", [](LoDTensorArray &self) { return self.size(); })
      .def("__setitem__",
           [](LoDTensorArray &self, size_t i, const LoDTensor &t) {
             PADDLE_ENFORCE_LT(i, self.size());
             self[i].ShareDataWith(t);
             self[i].set_lod(t.lod());
           })
      .def("append", [](LoDTensorArray &self, const LoDTensor &t) {
        self.emplace_back();
        self.back().ShareDataWith(t);
        self.back().set_lod(t.lod());
      });

  m.def("op_support_gpu", OpSupportGPU);
#ifdef PADDLE_WITH_CUDA
  m.def("get_cuda_device_count", platform::GetCUDADeviceCount);

  m.def("nvprof_init", platform::CudaProfilerInit);
  m.def("nvprof_start", platform::CudaProfilerStart);
  m.def("nvprof_stop", platform::CudaProfilerStop);
#endif

  py::enum_<platform::ProfilerState>(m, "ProfilerState", py::arithmetic())
      .value("kDisabled", platform::ProfilerState::kDisabled)
      .value("kCPU", platform::ProfilerState::kCPU)
      .value("kCUDA", platform::ProfilerState::kCUDA)
      .export_values();

  py::enum_<platform::EventSortingKey>(m, "EventSortingKey", py::arithmetic())
      .value("kDefault", platform::EventSortingKey::kDefault)
      .value("kCalls", platform::EventSortingKey::kCalls)
      .value("kTotal", platform::EventSortingKey::kTotal)
      .value("kMin", platform::EventSortingKey::kMin)
      .value("kMax", platform::EventSortingKey::kMax)
      .value("kAve", platform::EventSortingKey::kAve)
      .export_values();

  m.def("enable_profiler", platform::EnableProfiler);
  m.def("disable_profiler", platform::DisableProfiler);
  m.def("reset_profiler", platform::ResetProfiler);
  return m.ptr();
}
}  // namespace pybind
}  // namespace paddle
