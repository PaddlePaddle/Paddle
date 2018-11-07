/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include <Python.h>
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>  // NOLINT // for call_once
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/pass_builder.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/parallel_executor.h"
#include "paddle/fluid/framework/prune.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/pybind/const_value.h"
#include "paddle/fluid/pybind/exception.h"
#include "paddle/fluid/pybind/protobuf.h"
#include "paddle/fluid/pybind/pybind.h"  // NOLINT
#include "paddle/fluid/pybind/recordio.h"
#include "paddle/fluid/pybind/tensor_py.h"

#include "paddle/fluid/string/to_string.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/operators/nccl/nccl_gpu_common.h"
#include "paddle/fluid/platform/cuda_profiler.h"
#include "paddle/fluid/platform/gpu_info.h"
#endif

#include "pybind11/stl.h"

DEFINE_bool(reader_queue_speed_test_mode, false,
            "If set true, the queue.pop will only get data from queue but not "
            "remove the data from queue for speed testing");

// disable auto conversion to list in Python
PYBIND11_MAKE_OPAQUE(paddle::framework::LoDTensorArray);

namespace paddle {
namespace pybind {
bool IsCompiledWithCUDA() {
#ifndef PADDLE_WITH_CUDA
  return false;
#else
  return true;
#endif
}

bool IsCompiledWithDIST() {
#ifdef PADDLE_WITH_DISTRIBUTE
  return true;
#else
  return false;
#endif
}

PYBIND11_PLUGIN(core) {
  py::module m("core", "C++ core of PaddlePaddle");

  // using framework in this function. Since it is inside a function, it will
  // not cause namespace pollution.
  using namespace paddle::framework;  // NOLINT

  BindException(&m);

  py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
      .def_buffer(
          [](Tensor &self) -> py::buffer_info { return CastToPyBuffer(self); })
      .def("_get_dims",
           [](const Tensor &self) { return vectorize(self.dims()); })
      .def("_set_dims",
           [](Tensor &self, const std::vector<int64_t> &dim) {
             self.Resize(make_ddim(dim));
           })
      .def("_set_layout",
           [](Tensor &self, const std::string &layout) {
             self.set_layout(StringToDataLayout(layout));
           })
      .def("_alloc_float",
           [](Tensor &self, paddle::platform::CUDAPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_float",
           [](Tensor &self, paddle::platform::CPUPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("_alloc_int",
           [](Tensor &self, paddle::platform::CPUPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](Tensor &self, paddle::platform::CUDAPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_int",
           [](Tensor &self, paddle::platform::CUDAPinnedPlace &place) {
             self.mutable_data<int>(place);
           })
      .def("_alloc_float",
           [](Tensor &self, paddle::platform::CUDAPinnedPlace &place) {
             self.mutable_data<float>(place);
           })
      .def("set", PyCPUTensorSetFromArray<float>)
      .def("set", PyCPUTensorSetFromArray<int>)
      .def("set", PyCPUTensorSetFromArray<double>)
      .def("set", PyCPUTensorSetFromArray<int64_t>)
      .def("set", PyCPUTensorSetFromArray<bool>)
      .def("set", PyCPUTensorSetFromArray<uint16_t>)
      .def("set", PyCPUTensorSetFromArray<uint8_t>)
      .def("set", PyCPUTensorSetFromArray<int8_t>)
#ifdef PADDLE_WITH_CUDA
      .def("set", PyCUDATensorSetFromArray<float>)
      .def("set", PyCUDATensorSetFromArray<int>)
      .def("set", PyCUDATensorSetFromArray<double>)
      .def("set", PyCUDATensorSetFromArray<int64_t>)
      .def("set", PyCUDATensorSetFromArray<bool>)
      .def("set", PyCUDATensorSetFromArray<uint16_t>)
      .def("set", PyCUDATensorSetFromArray<uint8_t>)
      .def("set", PyCUDATensorSetFromArray<int8_t>)
      .def("set", PyCUDAPinnedTensorSetFromArray<float>)
      .def("set", PyCUDAPinnedTensorSetFromArray<int>)
      .def("set", PyCUDAPinnedTensorSetFromArray<double>)
      .def("set", PyCUDAPinnedTensorSetFromArray<int64_t>)
      .def("set", PyCUDAPinnedTensorSetFromArray<bool>)
      .def("set", PyCUDAPinnedTensorSetFromArray<uint16_t>)
      .def("set", PyCUDAPinnedTensorSetFromArray<uint8_t>)
      .def("set", PyCUDAPinnedTensorSetFromArray<int8_t>)
#endif
      .def("shape", [](Tensor &self) { return vectorize(self.dims()); })
      .def("_set_float_element", TensorSetElement<float>)
      .def("_get_float_element", TensorGetElement<float>)
      .def("_set_double_element", TensorSetElement<double>)
      .def("_get_double_element", TensorGetElement<double>)
      .def("_dtype", [](Tensor &self) { return ToDataType(self.type()); });

  py::class_<LoDTensor, Tensor>(m, "LoDTensor", R"DOC(
    LoDTensor is a Tensor with optional LoD information.

    np.array(lod_tensor) can convert LoDTensor to numpy array.
    lod_tensor.lod() can retrieve the LoD information.

    LoD is short for Level of Details and is usually used for varied sequence
    length. You can skip the following comment if you don't need optional LoD.

  For example:
     A LoDTensor X can look like the example below. It contains 2 sequences.
     The first has length 2 and the second has length 3, as described by x.lod.

     The first tensor dimension 5=2+3 is calculated from LoD if it's available.
     It means the total number of sequence element. In X, each element has 2
     columns, hence [5, 2].

      x.lod  = [[2, 3]]
      x.data = [[1, 2], [3, 4],
                [5, 6], [7, 8], [9, 10]]
      x.shape = [5, 2]

      LoD can have multiple levels (for example, a paragraph can have multiple
      sentences and a sentence can have multiple words). In the following
      LodTensor Y, the lod_level is 2. It means there are 2 sequence, the
      first sequence length is 2 (has 2 sub-sequences), the second one's
      length is 1. The first sequence's 2 sub-sequences have length 2 and 2,
      respectively. And the second sequence's 1 sub-sequence has length 3.

      y.lod = [[2 1], [2 2 3]]
      y.shape = [2+2+3, ...]

  Note:
      In above description, LoD is length-based. In Paddle internal
      implementation, lod is offset-based. Hence, internally,
      y.lod is represented as [[0, 2, 3], [0, 2, 4, 7]] (length-based
      equivlent would be [[2-0, 3-2], [2-0, 4-2, 7-4]]).

      Sometimes LoD is called recursive_sequence_length to be more
      self-explanatory. In this case, it must be length-based. Due to history
      reasons. when LoD is called lod in public API, it might be offset-based.
      Users should be careful about it.

        )DOC")
      .def_buffer(
          [](Tensor &self) -> py::buffer_info { return CastToPyBuffer(self); })
      .def("__init__",
           [](LoDTensor &instance, const std::vector<std::vector<size_t>>
                                       &recursive_sequence_lengths) {
             LoD new_lod;
             new_lod.reserve(recursive_sequence_lengths.size());
             std::copy(recursive_sequence_lengths.begin(),
                       recursive_sequence_lengths.end(),
                       std::back_inserter(new_lod));
             LoD new_offset_lod = ConvertToOffsetBasedLoD(new_lod);
             PADDLE_ENFORCE(
                 CheckLoD(new_offset_lod, -1),
                 "the provided recursive_sequence_lengths info is invalid");
             new (&instance) LoDTensor(new_offset_lod);
           })
      .def("__init__", [](LoDTensor &instance) { new (&instance) LoDTensor(); })
      // We implement offset based LOD in C++ while we use length based with
      // Python API. So we changed set_lod to set_recursive_sequence_lengths to
      // avoid misuse.
      // The discussion is here:
      // https://github.com/PaddlePaddle/Paddle/issues/10855
      .def("set_lod",
           [](LoDTensor &self, const std::vector<std::vector<size_t>> &lod) {
             // the input lod is offset-based level-of-detail info
             LoD new_lod;
             new_lod.reserve(lod.size());
             std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
             PADDLE_ENFORCE(CheckLoD(new_lod, vectorize(self.dims()).front()),
                            "the provided lod info is invalid");
             self.set_lod(new_lod);
           })
      .def("set_recursive_sequence_lengths",
           [](LoDTensor &self, const std::vector<std::vector<size_t>>
                                   &recursive_sequence_lengths) {
             // the input recursive_sequence_lengths is length-based
             // level-of-detail info
             LoD new_lod;
             new_lod.reserve(recursive_sequence_lengths.size());
             std::copy(recursive_sequence_lengths.begin(),
                       recursive_sequence_lengths.end(),
                       std::back_inserter(new_lod));
             LoD new_offset_lod = ConvertToOffsetBasedLoD(new_lod);
             PADDLE_ENFORCE(
                 CheckLoD(new_offset_lod, vectorize(self.dims()).front()),
                 "the provided recursive_sequence_lengths info is invalid");
             self.set_lod(new_offset_lod);
           })
      .def("lod",
           [](LoDTensor &self) -> std::vector<std::vector<size_t>> {
             // output the offset-based lod info
             LoD lod = self.lod();
             std::vector<std::vector<size_t>> new_lod;
             new_lod.reserve(lod.size());
             std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
             return new_lod;
           })
      // Set above comments of set_lod.
      .def("recursive_sequence_lengths",
           [](LoDTensor &self) -> std::vector<std::vector<size_t>> {
             // output the length-based lod info
             LoD lod = ConvertToLengthBasedLoD(self.lod());
             std::vector<std::vector<size_t>> new_lod;
             new_lod.reserve(lod.size());
             std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod));
             return new_lod;
           })
      .def("has_valid_recursive_sequence_lengths", [](LoDTensor &self) -> bool {
        // Check that the lod info is valid and match the outermost
        // dimension of the LoDTensor data
        return CheckLoD(self.lod(), vectorize(self.dims()).front());
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
      .def("sync_index", [](SelectedRows &instance) { instance.SyncIndex(); })
      .def("rows", [](SelectedRows &self) {
        auto rows = self.rows();
        std::vector<int64_t> new_rows;
        new_rows.reserve(rows.size());
        std::copy(rows.begin(), rows.end(), std::back_inserter(new_rows));
        return new_rows;
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
      .def("get_reader",
           [](Variable &self) -> framework::ReaderHolder * {
             PADDLE_ENFORCE(self.IsType<framework::ReaderHolder>());
             return self.GetMutable<framework::ReaderHolder>();
           },
           py::return_value_policy::reference);

  py::class_<framework::ReaderHolder>(m, "Reader", "")
      .def("reset", &framework::ReaderHolder::ResetAll);

  using LoDTensorBlockingQueue =
      ::paddle::operators::reader::LoDTensorBlockingQueue;
  using LoDTensorBlockingQueueHolder =
      ::paddle::operators::reader::LoDTensorBlockingQueueHolder;
  py::class_<LoDTensorBlockingQueue, std::shared_ptr<LoDTensorBlockingQueue>>(
      m, "LoDTensorBlockingQueue", "")
      .def("push",
           [](LoDTensorBlockingQueue &self,
              const std::vector<framework::LoDTensor> &lod_tensor_vec) {
             pybind11::gil_scoped_release release;
             return self.Push(lod_tensor_vec);
           })
      .def("size", &LoDTensorBlockingQueue::Size)
      .def("capacity", &LoDTensorBlockingQueue::Cap)
      .def("close", &LoDTensorBlockingQueue::Close)
      .def("is_closed", &LoDTensorBlockingQueue::IsClosed);

  m.def("init_lod_tensor_blocking_queue",
        [](Variable &var, size_t capacity,
           const std::vector<std::vector<int64_t>> &shapes)
            -> std::shared_ptr<LoDTensorBlockingQueue> {
              std::vector<DDim> dims(shapes.size());
              std::transform(shapes.begin(), shapes.end(), dims.begin(),
                             [](const std::vector<int64_t> &shape) {
                               return make_ddim(shape);
                             });
              auto *holder = var.GetMutable<LoDTensorBlockingQueueHolder>();
              holder->InitOnce(capacity, dims,
                               FLAGS_reader_queue_speed_test_mode);
              return holder->GetQueue();
            },
        py::return_value_policy::copy);

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
      prog_with_targets.MutableBlock(t[0])->Op(t[1])->SetIsTarget(true);
    }
    proto::ProgramDesc pruned_desc;
    Prune(*prog_with_targets.Proto(), &pruned_desc);
    return new ProgramDesc(pruned_desc);
  });
  m.def("empty_var_name",
        []() { return std::string(framework::kEmptyVarName); });
  m.def("grad_var_suffix",
        []() { return std::string(framework::kGradVarSuffix); });
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
                  })
          .def_static("create",
                [](paddle::platform::CUDAPinnedPlace& place)
                        -> paddle::platform::DeviceContext* {
#ifndef PADDLE_WITH_CUDA
                  PADDLE_THROW(
                        "CUDAPinnedPlace is not supported in CPU device.");
#else
                  return new paddle::platform::CUDAPinnedDeviceContext(place);
#endif
                });;
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

  py::class_<paddle::platform::CUDAPinnedPlace>(m, "CUDAPinnedPlace")
      .def(py::init<>())
      .def("__str__", string::to_string<const platform::CUDAPinnedPlace &>);

  py::class_<platform::Place>(m, "Place")
      .def(py::init<>())
      .def("set_place",
           [](platform::Place &self, const platform::CPUPlace &cpu_place) {
             self = cpu_place;
           })
      .def("set_place",
           [](platform::Place &self, const platform::CUDAPlace &gpu_place) {
             self = gpu_place;
           })
      .def("set_place", [](platform::Place &self,
                           const platform::CUDAPinnedPlace &cuda_pinned_place) {
        self = cuda_pinned_place;
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
      .def("run",
           [](OperatorBase &self, const Scope &scope,
              const platform::CPUPlace &place) { self.Run(scope, place); })
      .def("run",
           [](OperatorBase &self, const Scope &scope,
              const platform::CUDAPlace &place) { self.Run(scope, place); })
      .def("run",
           [](OperatorBase &self, const Scope &scope,
              const platform::CUDAPinnedPlace &place) {
             self.Run(scope, place);
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

  py::class_<framework::Executor>(m, "Executor")
      .def(py::init<const platform::Place &>())
      .def("close", &Executor::Close)
      .def("run", [](Executor &self, const ProgramDesc &prog, Scope *scope,
                     int block_id, bool create_local_scope, bool create_vars) {
        pybind11::gil_scoped_release release;
        self.Run(prog, scope, block_id, create_local_scope, create_vars);
      });

  m.def("init_gflags", framework::InitGflags);
  m.def("init_glog", framework::InitGLOG);
  m.def("init_devices",
        [](bool init_p2p) { framework::InitDevices(init_p2p); });

  m.def("is_compiled_with_cuda", IsCompiledWithCUDA);
  m.def("is_compiled_with_dist", IsCompiledWithDIST);
#ifdef PADDLE_WITH_CUDA
  m.def("is_float16_supported", [](const platform::CUDAPlace &place) -> bool {
    // Only GPUs with Compute Capability >= 53 support float16
    return platform::GetCUDAComputeCapability(place.device) >= 53;
  });
#endif

  m.def("set_feed_variable", framework::SetFeedVariable);
  m.def("get_fetch_variable", framework::GetFetchVariable);

  m.def("_is_program_version_supported", IsProgramVersionSupported);

  BindProgramDesc(&m);
  BindBlockDesc(&m);
  BindVarDsec(&m);
  BindOpDesc(&m);
  BindConstValue(&m);

  py::class_<framework::LoDRankTable>(m, "LodRankTable")
      .def("items", [](framework::LoDRankTable &table) {
        std::vector<std::pair<size_t, size_t>> res;
        for (auto &item : table.items()) {
          res.push_back({item.index, item.length});
        }
        return res;
      });

  py::class_<LoDTensorArray>(m, "LoDTensorArray")
      .def("__init__",
           [](LoDTensorArray &instance) { new (&instance) LoDTensorArray(); })
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

  m.def("IsInplace",
        [](std::string op) -> bool { return operators::IsInplace(op); });

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
      .value("kAll", platform::ProfilerState::kAll)
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
  m.def("is_profiler_enabled", platform::IsProfileEnabled);
  m.def("reset_profiler", platform::ResetProfiler);

  py::class_<ir::Pass, std::shared_ptr<ir::Pass>> pass(m, "Pass");
  pass.def(py::init())
      .def(
          "set_str",
          [](ir::Pass &self, const std::string &name, const std::string &attr) {
            self.Set<std::string>(name, new std::string(attr));
          })
      .def("set_int", [](ir::Pass &self, const std::string &name, int val) {
        self.Set<const int>(name, new int(val));
      });

  py::class_<ir::PassBuilder, std::shared_ptr<ir::PassBuilder>> pb(
      m, "PassBuilder");
  pb.def(py::init())
      .def("append_pass",
           [](ir::PassBuilder &self,
              const std::string &pass_type) -> std::shared_ptr<ir::Pass> {
             return self.AppendPass(pass_type);
           })
      .def("all_passes", [](ir::PassBuilder &self) { return self.AllPasses(); })
      .def("insert_pass",
           [](ir::PassBuilder &self, size_t idx, const std::string &pass_type) {
             return self.InsertPass(idx, pass_type);
           })
      .def("remove_pass",
           [](ir::PassBuilder &self, size_t idx) { self.RemovePass(idx); });

  // -- python binds for parallel executor.
  py::class_<ParallelExecutor> pe(m, "ParallelExecutor");
  py::class_<ExecutionStrategy> exec_strategy(pe, "ExecutionStrategy", R"DOC(
    ExecutionStrategy allows the user to more preciously control how to run
    the program in ParallelExecutor by setting the property.

    Examples:
        .. code-block:: python

          exec_strategy = fluid.ExecutionStrategy()
          exec_strategy.num_threads = 4

          train_exe = fluid.ParallelExecutor(use_cuda=True,
                                             loss_name=loss.name,
                                             exec_strategy=exec_strategy)

          train_loss, = train_exe.run([loss.name], feed=feed_dict)

        )DOC");

  exec_strategy.def(py::init())
      .def_property(
          "num_threads",
          [](const ExecutionStrategy &self) { return self.num_threads_; },
          [](ExecutionStrategy &self, size_t num_threads) {
            self.num_threads_ = num_threads;
          },
          R"DOC(The type is INT, num_threads represents the size of thread pool that
            used to run the operators of the current program in ParallelExecutor.
            If :math:`num\_threads=1`, all the operators will execute one by one,
            but the order maybe difference between iterations.
            If it is not set, it will be set in ParallelExecutor according to the
            device type and device count, for GPU, :math:`num\_threads=device\_count*4`, for CPU,
            :math:`num\_threads=CPU\_NUM*4`, the explanation of:math:`CPU\_NUM` is in ParallelExecutor.
            if it is not set, ParallelExecutor will get the cpu count by calling
            `multiprocessing.cpu_count()`. Default 0.)DOC")
      .def_property(
          "use_cuda",
          [](const ExecutionStrategy &self) { return self.use_cuda_; },
          [](ExecutionStrategy &self, bool use_cuda) {
            self.use_cuda_ = use_cuda;
          })  // FIXME(chengduo): Doesn't add doc for 'use_cuda', use_cuda may
      // make user confuse, because ParallelExecutor has a parameter named
      // 'use_cuda' too, in current implementation, ParallelExecutor's
      // 'use_cuda' will rewrite ExecutionStrategy's 'use_cuda'.
      .def_property(
          "allow_op_delay",
          [](const ExecutionStrategy &self) { return self.allow_op_delay_; },
          [](ExecutionStrategy &self, bool allow_op_delay) {
            self.allow_op_delay_ = allow_op_delay;
          },
          R"DOC(The type is BOOL, allow_op_delay represents whether to delay the
                communication operators to run, it may make the execution faster.
                Note that in some models, allow_op_delay may cause program hang. Default False.)DOC")
      .def_property(
          "num_iteration_per_drop_scope",
          [](const ExecutionStrategy &self) {
            return self.num_iteration_per_drop_scope_;
          },
          [](ExecutionStrategy &self, size_t num_iteration_per_drop_scope) {
            self.num_iteration_per_drop_scope_ = num_iteration_per_drop_scope;
          },
          R"DOC(The type is INT, num_iteration_per_drop_scope indicates how
                many iterations to clean up the temp variables which
                is generated during execution. It may make the execution faster,
                because the temp variable's shape maybe the same between two iterations. Default 100.

                NOTES:
                    1. If you fetch data when calling the 'run', the ParallelExecutor
                       will clean up the temp variables at the end of the current iteration.
                    2. In some NLP model, it may cause the GPU memory is insufficient,
                       in this case, you should reduce `num_iteration_per_drop_scope`.
              )DOC");

  exec_strategy.def_property(
      "use_experimental_executor",
      [](const ExecutionStrategy &self) {
        return self.type_ == ExecutionStrategy::kExperimental;
      },
      [](ExecutionStrategy &self, bool experimental) {
        self.type_ = experimental ? ExecutionStrategy::kExperimental
                                  : ExecutionStrategy::kDefault;
      });

  py::class_<BuildStrategy> build_strategy(pe, "BuildStrategy", R"DOC(
    BuildStrategy allows the user to more preciously control how to
    build the SSA Graph in ParallelExecutor by setting the property.

    Examples:
        .. code-block:: python

          build_strategy = fluid.BuildStrategy()
          build_strategy.reduce_strategy = fluid.BuildStrategy.ReduceStrategy.Reduce

          train_exe = fluid.ParallelExecutor(use_cuda=True,
                                             loss_name=loss.name,
                                             build_strategy=build_strategy)

          train_loss, = train_exe.run([loss.name], feed=feed_dict)
)DOC");

  py::enum_<BuildStrategy::ReduceStrategy>(build_strategy, "ReduceStrategy")
      .value("Reduce", BuildStrategy::ReduceStrategy::kReduce)
      .value("AllReduce", BuildStrategy::ReduceStrategy::kAllReduce);
  py::enum_<BuildStrategy::GradientScaleStrategy>(build_strategy,
                                                  "GradientScaleStrategy")
      .value("CoeffNumDevice",
             BuildStrategy::GradientScaleStrategy::kCoeffNumDevice)
      .value("One", BuildStrategy::GradientScaleStrategy::kOne)
      .value("Customized", BuildStrategy::GradientScaleStrategy::kCustomized);

  build_strategy.def(py::init())
      .def_property(
          "reduce_strategy",
          [](const BuildStrategy &self) { return self.reduce_; },
          [](BuildStrategy &self, BuildStrategy::ReduceStrategy strategy) {
            self.reduce_ = strategy;
          },
          R"DOC(The type is STR, there are two reduce strategies in ParallelExecutor,
                  'AllReduce' and 'Reduce'. If you want that all the parameters'
                  optimization are done on all devices independently, you should choose 'AllReduce';
                  if you choose 'Reduce', all the parameters' optimization will be evenly distributed
                  to different devices, and then broadcast the optimized parameter to other devices.
                  In some models, `Reduce` is faster. Default 'AllReduce'. )DOC")
      .def_property(
          "gradient_scale_strategy",
          [](const BuildStrategy &self) { return self.gradient_scale_; },
          [](BuildStrategy &self,
             BuildStrategy::GradientScaleStrategy strategy) {
            self.gradient_scale_ = strategy;
          },
          R"DOC(The type is STR, there are three ways of defining :math:`loss@grad` in
                   ParallelExecutor, 'CoeffNumDevice', 'One' and 'Customized'. By default,
                   ParallelExecutor sets the :math:`loss@grad` according to the number of devices.
                   If you want to customize :math:`loss@grad`, you can choose 'Customized'.
                   Default 'CoeffNumDevice'.)DOC")
      .def_property(
          "debug_graphviz_path",
          [](const BuildStrategy &self) { return self.debug_graphviz_path_; },
          [](BuildStrategy &self, const std::string &path) {
            self.debug_graphviz_path_ = path;
          },
          R"DOC(The type is STR, debug_graphviz_path indicate the path that
                    writing the SSA Graph to file in the form of graphviz, you.
                    It is useful for debugging. Default "")DOC")
      .def_property(
          "enable_data_balance",
          [](const BuildStrategy &self) { return self.enable_data_balance_; },
          [](BuildStrategy &self, bool b) {
            self.enable_data_balance_ = b;
          })  // FIXME(chengudo): enable_data_balance seems not important
      .def_property(
          "enable_sequential_execution",
          [](const BuildStrategy &self) {
            return self.enable_sequential_execution_;
          },
          [](BuildStrategy &self, bool b) {
            self.enable_sequential_execution_ = b;
          },
          R"DOC(The type is BOOL. If set True, the execution order of ops would be the same as what is in the program. Default False.)DOC")
      .def_property(
          "remove_unnecessary_lock",
          [](const BuildStrategy &self) {
            return self.remove_unnecessary_lock_;
          },
          [](BuildStrategy &self, bool b) {
            self.remove_unnecessary_lock_ = b;
          },
          R"DOC(The type is BOOL. If set True, some locks in GPU ops would be released and ParallelExecutor would run faster. Default False.)DOC")
      .def_property(
          "fuse_elewise_add_act_ops",
          [](const BuildStrategy &self) {
            return self.fuse_elewise_add_act_ops_;
          },
          [](BuildStrategy &self, bool b) {
            self.fuse_elewise_add_act_ops_ = b;
          },
          R"DOC(The type is BOOL, fuse_elewise_add_act_ops indicate whether
                     to fuse elementwise_add_op and activation_op,
                     it may make the execution faster. Default False)DOC")
      .def("_create_passes_from_strategy",
           [](BuildStrategy &self) -> std::shared_ptr<ir::PassBuilder> {
             return self.CreatePassesFromStrategy();
           });

  pe.def(py::init<const std::vector<platform::Place> &,
                  const std::unordered_set<std::string> &,
                  const std::unordered_set<std::string> &, const ProgramDesc &,
                  const std::string &, Scope *, std::vector<Scope *> &,
                  const ExecutionStrategy &, const BuildStrategy &, size_t,
                  size_t>())
      // NOTE: even we return a vec<Scope*>* to Python use reference policy.
      // We still cannot get local_scope from this vector, since the element
      // of vec<Scope*> will be freed by Python GC. We can only return Scope*
      // one by one and mark them as reference.
      .def("local_scopes",
           [](ParallelExecutor &self) -> std::vector<Scope *> * {
             return &self.GetLocalScopes();
           },
           py::return_value_policy::reference)
      .def("feed_tensors_into_local_scopes",
           &ParallelExecutor::FeedTensorsIntoLocalScopes)
      .def("feed_and_split_tensor_into_local_scopes",
           &ParallelExecutor::FeedAndSplitTensorIntoLocalScopes)
      .def("run", [](ParallelExecutor &self,
                     const std::vector<std::string> &fetch_tensors,
                     const std::string &fetched_var_name) {
        pybind11::gil_scoped_release release;
        self.Run(fetch_tensors, fetched_var_name);
      });

  BindRecordIOWriter(&m);
  return m.ptr();
}
}  // namespace pybind
}  // namespace paddle
