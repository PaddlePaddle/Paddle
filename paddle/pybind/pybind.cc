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
#include "gflags/gflags.h"
#include "paddle/framework/backward.h"
#include "paddle/framework/executor.h"
#include "paddle/framework/feed_fetch_method.h"
#include "paddle/framework/framework.pb.h"
#include "paddle/framework/lod_rank_table.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/lod_tensor_array.h"
#include "paddle/framework/prune.h"
#include "paddle/framework/selected_rows.h"
#include "paddle/framework/tensor_array.h"
#include "paddle/operators/cond_op.h"
#include "paddle/operators/dynamic_recurrent_op.h"
#include "paddle/operators/net_op.h"
#include "paddle/platform/enforce.h"
#include "paddle/platform/place.h"
#include "paddle/pybind/exception.h"
#include "paddle/pybind/pybind.h"
#include "paddle/pybind/tensor_py.h"
#include "paddle/string/to_string.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/operators/nccl/nccl_gpu_common.h"
#include "paddle/platform/gpu_info.h"
#endif

namespace paddle {
namespace pybind {
static size_t UniqueIntegerGenerator(const std::string &prefix) {
  static std::unordered_map<std::string, std::atomic<size_t>> generators;
  return generators[prefix].fetch_add(1);
}

std::once_flag gflags_init_flag;

// TODO(qijun) move init gflags to init.cc
void InitGflags(std::vector<std::string> &argv) {
  std::call_once(gflags_init_flag, [&]() {
    int argc = argv.size();
    char **arr = new char *[argv.size()];
    std::string line;
    for (size_t i = 0; i < argv.size(); i++) {
      arr[i] = &argv[i][0];
      line += argv[i];
      line += ' ';
    }
    google::ParseCommandLineFlags(&argc, &arr, true);
    VLOG(1) << "Init commandline: " << line;
  });
}

bool IsCompileGPU() {
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
      .def("alloc_float",
           [](Tensor &self, paddle::platform::GPUPlace &place) {
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
           [](Tensor &self, paddle::platform::GPUPlace &place) {
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
#ifndef PADDLE_WITH_CUDA
            new (&instance) LoDTensor(lod);
#else
             LoD new_lod;
             new_lod.reserve(lod.size());
             std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
             new (&instance) LoDTensor(new_lod);
#endif
          })
      .def("__init__", [](LoDTensor &instance) { new (&instance) LoDTensor(); })
      .def("set_lod",
           [](LoDTensor &self, const std::vector<std::vector<size_t>> &lod) {
#ifndef PADDLE_WITH_CUDA
             self.set_lod(lod);
#else
             LoD new_lod;
             new_lod.reserve(lod.size());
             std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
             self.set_lod(new_lod);
#endif
           })
      .def("lod", [](LoDTensor &self) -> std::vector<std::vector<size_t>> {
#ifndef PADDLE_WITH_CUDA
        return self.lod();
#else
           auto lod = self.lod();
           std::vector<std::vector<size_t>> new_lod;
           new_lod.reserve(lod.size());
           std::transform(lod.begin(), lod.end(), std::back_inserter(new_lod),
               [](Vector<size_t> item) ->
                   std::vector<size_t> {
                 std::vector<size_t> v;
                 v.reserve(item.size());
                 std::copy(item.begin(), item.end(), std::back_inserter(v));
                 return v;
               });
           return new_lod;
#endif
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
  m.def("prune", [](const ProgramDescBind &origin,
                    const std::vector<std::array<size_t, 2>> &targets) {
    ProgramDescBind prog_with_targets(origin);
    for (const auto &t : targets) {
      prog_with_targets.MutableBlock(t[0])->Op(t[1])->MarkAsTarget();
    }
    ProgramDesc pruned_desc;
    Prune(*prog_with_targets.Proto(), &pruned_desc);
    return new ProgramDescBind(pruned_desc);
  });
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
                  [](paddle::platform::GPUPlace& place)
                      -> paddle::platform::DeviceContext* {
#ifndef PADDLE_WITH_CUDA
                    PADDLE_THROW("GPUPlace is not supported in CPU device.");
#else
                    return new paddle::platform::CUDADeviceContext(place);
#endif
                  });
// clang-format on

#ifdef PADDLE_WITH_CUDA
  py::class_<platform::Communicator>(m, "Communicator").def(py::init<>());
#endif
  py::class_<platform::GPUPlace>(m, "GPUPlace")
      .def(py::init<int>())
      .def("__str__", string::to_string<const platform::GPUPlace &>);

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
           [](platform::Place &self, const platform::GPUPlace &gpu_place) {
             self = gpu_place;
           });

  py::class_<OperatorBase>(m, "Operator")
      .def_static("create",
                  [](py::bytes protobin) {
                    OpDesc desc;
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
              const platform::DeviceContext &dev_ctx) {
             self.Run(scope, dev_ctx);
             dev_ctx.Wait();
           })
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

  py::class_<framework::TensorArray>(m, "TensorArray")
      .def("__init__",
           [](TensorArray &instance) { new (&instance) TensorArray(); })
      .def("read",
           [](TensorArray &self, size_t index) { return self.Read(index); })
      .def("write", [](TensorArray &self, size_t index,
                       LoDTensor &value) { self.Write(index, value); })
      .def("write_shared",
           [](TensorArray &self, size_t index, const LoDTensor &value) {
             self.WriteShared(index, value);
           })
      .def("size", [](TensorArray &self) { return self.size(); })
      .def("pack",
           [](TensorArray &self, size_t level,
              const std::vector<std::vector<size_t>> &meta_info,
              const std::vector<std::vector<size_t>> &lod) {
             std::vector<DySeqMeta> meta;
             for (auto &info : meta_info) {
               PADDLE_ENFORCE_EQ(info.size(), 3UL);
               meta.emplace_back(info[0], info[1], info[2]);
             }
#ifndef PADDLE_WITH_CUDA
             return self.Pack(level, meta, lod);
#else
             LoD new_lod;
             new_lod.reserve(lod.size());
             std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
             return self.Pack(level, meta, new_lod);
#endif
           })
      .def("unpack",
           [](TensorArray &self, const LoDTensor &source, int level,
              bool length_descend) {
             auto metas = self.Unpack(source, level, length_descend);
             std::vector<std::vector<size_t>> meta_info;
             for (auto meta : metas) {
               meta_info.emplace_back(
                   std::vector<size_t>({meta.begin, meta.end, meta.ori_idx}));
             }
             return meta_info;
           })
      .def("stack", [](TensorArray &self) { return self.Stack(); })
      .def("unstack",
           [](TensorArray &self, const LoDTensor &source) {
             return self.Unstack(source);
           })
      .def("unstack_shared", [](TensorArray &self, const LoDTensor &source) {
        return self.UnstackShared(source);
      });

  py::class_<operators::DynamicRecurrentOp, OperatorBase>(m,
                                                          "DynamicRecurrentOp")
      .def_static("create",
                  [](py::bytes protobin) -> operators::DynamicRecurrentOp * {
                    OpDesc desc;
                    PADDLE_ENFORCE(desc.ParsePartialFromString(protobin),
                                   "Cannot parse user input to OpDesc");
                    PADDLE_ENFORCE(desc.IsInitialized(),
                                   "User OpDesc is not initialized, reason %s",
                                   desc.InitializationErrorString());
                    auto rnn_op = OpRegistry::CreateOp(desc);
                    return static_cast<operators::DynamicRecurrentOp *>(
                        rnn_op.release());
                  })
      .def("set_step_unit",
           [](operators::DynamicRecurrentOp &self, const operators::NetOp &net)
               -> void { self.rnn.SetStepUnit(net.Clone()); })
      .def("get_state",
           [](operators::DynamicRecurrentOp &self, const std::string &name)
               -> const TensorArray & { return self.rnn.state(name); })
      .def("get_step_input",
           [](operators::DynamicRecurrentOp &self, const std::string &name)
               -> const TensorArray & { return self.rnn.step_input(name); })
      .def("get_step_output",
           [](operators::DynamicRecurrentOp &self, const std::string &name)
               -> const TensorArray & { return self.rnn.step_output(name); });

  // cond_op
  py::class_<operators::CondOp, OperatorBase>(m, "CondOp")
      .def_static("create",
                  [](py::bytes protobin) -> operators::CondOp * {
                    OpDesc desc;
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
      .def(py::init<std::vector<platform::Place> &>())
      .def("run", &Executor::Run);

  m.def("unique_integer", UniqueIntegerGenerator);
  m.def("init_gflags", InitGflags);

  m.def("is_compile_gpu", IsCompileGPU);
  m.def("set_feed_variable", framework::SetFeedVariable);
  m.def("get_fetch_variable", framework::GetFetchVariable);

  BindProgramDesc(m);
  BindBlockDesc(m);
  BindVarDsec(m);
  BindOpDesc(m);

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
#endif

  return m.ptr();
}
}  // namespace pybind
}  // namespace paddle
