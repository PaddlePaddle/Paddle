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

#include <Python.h>
#include <algorithm>
#include <map>
#include <memory>
#include <mutex>  // NOLINT // for call_once
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

//#include "paddle/fluid/framework/executor.h"
//#include "paddle/fluid/framework/feed_fetch_method.h"
//#include "paddle/fluid/framework/framework.pb.h"
//#include "paddle/fluid/framework/ir/pass_builder.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
//#include "paddle/fluid/framework/parallel_executor.h"
//#include "paddle/fluid/framework/prune.h"
//#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/scope_pool.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/version.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/memory/allocation/allocator_strategy.h"
#include "paddle/fluid/memory/allocation/legacy_allocator.h"
//#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/py_func_op.h"
//#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"
//#include "paddle/fluid/platform/cpu_info.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/init.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/profiler.h"
#include "paddle/fluid/pybind/async_executor_py.h"
#include "paddle/fluid/pybind/const_value.h"
#include "paddle/fluid/tensor_test/exception.h"
//#include "paddle/fluid/pybind/imperative.h"
//#include "paddle/fluid/pybind/inference_api.h"
//#include "paddle/fluid/pybind/ir.h"
#include "paddle/fluid/pybind/protobuf.h"
#include "paddle/fluid/pybind/pybind.h"  // NOLINT
//#include "paddle/fluid/pybind/recordio.h"
#include "paddle/fluid/pybind/tensor_py.h"

#include "paddle/fluid/string/to_string.h"

#include "pybind11/stl.h"

// disable auto conversion to list in Python
PYBIND11_MAKE_OPAQUE(paddle::framework::LoDTensorArray);

namespace paddle {
namespace pybind {

PYBIND11_MODULE(test, m) {
  m.

      doc() = "C++ test of tensort";

  // using framework in this function. Since it is inside a function, it will
  // not cause namespace pollution.
  using namespace paddle::framework;  // NOLINT

  BindException(&m);
  py::class_<Tensor>(m, "Tensor",

                     py::buffer_protocol()

                         )
      .def_buffer(
          [](Tensor &self) -> py::buffer_info { return CastToPyBuffer(self); })
      .def("_get_dims",
           [](const Tensor &self) {
             return vectorize(self.

                              dims()

             );
           })
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
      .def("shape",
           [](Tensor &self) {
             return vectorize(self.

                              dims()

             );
           })
      .def("_set_float_element", TensorSetElement<float>)
      .def("_get_float_element", TensorGetElement<float>)
      .def("_set_double_element", TensorSetElement<double>)
      .def("_get_double_element", TensorGetElement<double>)
      .def("_place",
           [](Tensor &self) {
             return self.

                 place();
           })
      .def("_dtype", [](Tensor &self) {
        return self.

            type();
      })
      .def("__getitem__", PySliceTensor
        ,py::return_value_policy::reference);

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
             new_lod.reserve(recursive_sequence_lengths.

                             size()

             );
             std::copy(recursive_sequence_lengths.

                       begin(),
                       recursive_sequence_lengths

                           .

                       end(),
                       std::back_inserter(new_lod)

             );
             LoD new_offset_lod = ConvertToOffsetBasedLoD(new_lod);
             PADDLE_ENFORCE(
                 CheckLoD(new_offset_lod, -1),
                 "the provided recursive_sequence_lengths info is invalid");
             new (&instance) LoDTensor(new_offset_lod);
           })
      .def("__init__",
           [](LoDTensor &instance) {
             new (&instance)

                 LoDTensor();
           })
      // We implement offset based LOD in C++ while we use length based with
      // Python API. So we changed set_lod to set_recursive_sequence_lengths to
      // avoid misuse.
      // The discussion is here:
      // https://github.com/PaddlePaddle/Paddle/issues/10855
      .def("set_lod",
           [](LoDTensor &self, const std::vector<std::vector<size_t>> &lod) {
             // the input lod is offset-based level-of-detail info
             LoD new_lod;
             new_lod.reserve(lod.

                             size()

             );
             std::copy(lod.

                       begin(),
                       lod

                           .

                       end(),
                       std::back_inserter(new_lod)

             );
             PADDLE_ENFORCE(CheckLoD(new_lod, vectorize(self.dims()).front()),
                            "the provided lod info is invalid");
             self.set_lod(new_lod);
           },
           py::arg("lod"), R"DOC(
           Set LoD of the LoDTensor.

           Args:
               lod (List[List[int]]): the lod to be set.
           )DOC")
      .def("set_recursive_sequence_lengths",
           [](LoDTensor &self, const std::vector<std::vector<size_t>>
                                   &recursive_sequence_lengths) {
             // the input recursive_sequence_lengths is length-based
             // level-of-detail info
             LoD new_lod;
             new_lod.reserve(recursive_sequence_lengths.

                             size()

             );
             std::copy(recursive_sequence_lengths.

                       begin(),
                       recursive_sequence_lengths

                           .

                       end(),
                       std::back_inserter(new_lod)

             );
             LoD new_offset_lod = ConvertToOffsetBasedLoD(new_lod);
             PADDLE_ENFORCE(
                 CheckLoD(new_offset_lod, vectorize(self.dims()).front()),
                 "the provided recursive_sequence_lengths info is invalid");
             self.set_lod(new_offset_lod);
           },
           py::arg("recursive_sequence_lengths"), R"DOC(
           Set LoD of the LoDTensor according to recursive sequence length.

           For example, if recursive_sequence_lengths=[[2, 3]], meaning that
           there are two sequences with length 2 and 3 respectively, the
           corresponding lod would be [[0, 2, 2+3]], i.e, [[0, 2, 5]].

           Args:
                recursive_sequence_lengths (List[List[int]]): sequence lengths.
           )DOC")
      .def("lod",
           [](LoDTensor &self) -> std::vector<std::vector<size_t>> {
             // output the offset-based lod info
             LoD lod = self.lod();
             std::vector<std::vector<size_t>> new_lod;
             new_lod.reserve(lod.

                             size()

             );
             std::copy(lod.

                       begin(),
                       lod

                           .

                       end(),
                       std::back_inserter(new_lod)

             );
             return new_lod;
           },
           R"DOC(
           Return the LoD of the LoDTensor.

           Returns:
               out (List[List[int]]): the lod of the LoDTensor.
           )DOC")
      // Set above comments of set_lod.
      .def("recursive_sequence_lengths",
           [](LoDTensor &self) -> std::vector<std::vector<size_t>> {
             // output the length-based lod info
             LoD lod = ConvertToLengthBasedLoD(self.lod());
             std::vector<std::vector<size_t>> new_lod;
             new_lod.reserve(lod.size());
             std::copy(lod.begin(), lod.end(), std::back_inserter(new_lod)

             );
             return new_lod;
           },
           R"DOC(
           Return the sequence length of the LoDTensor corresponding to LoD.

           Returns:
               out (List[List[int]): the sequence lengths.
           )DOC")
      .def("has_valid_recursive_sequence_lengths",
           [](LoDTensor &self) -> bool {
             // Check that the lod info is valid and match the outermost
             // dimension of the LoDTensor data
             return CheckLoD(self.

                             lod(),
                             vectorize(self.dims())

                                 .

                             front()

             );
           },
           R"DOC(
           Check whether the lod of the LoDTensor is valid.

           Returns:
               out (bool): whether the lod is valid.
           )DOC");
}
}  // namespace pybind
}  // namespace paddle
