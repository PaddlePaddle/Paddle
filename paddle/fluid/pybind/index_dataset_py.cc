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

#include <fcntl.h>

#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include "paddle/fluid/pybind/index_dataset_py.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/distributed/index_dataset/index_sampler.h"
#include "paddle/fluid/distributed/index_dataset/index_wrapper.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

using paddle::distributed::TreeIndex;
using paddle::distributed::IndexWrapper;
using paddle::distributed::Node;

void BindIndexNode(py::module* m) {
  py::class_<Node>(*m, "IndexNode")
      .def(py::init<>())
      .def("id", [](Node& self) { return self.id(); })
      .def("is_leaf", [](Node& self) { return self.is_leaf(); })
      .def("probability", [](Node& self) { return self.probability(); });
}

void BindTreeIndex(py::module* m) {
  py::class_<TreeIndex, std::shared_ptr<TreeIndex>>(*m, "TreeIndex")
      .def(py::init([](const std::string name, const std::string path) {
        auto index_wrapper = IndexWrapper::GetInstancePtr();
        index_wrapper->insert_tree_index(name, path);
        return index_wrapper->GetTreeIndex(name);
      }))
      .def("height", [](TreeIndex& self) { return self.height(); })
      .def("branch", [](TreeIndex& self) { return self.branch(); })
      .def("total_node_nums",
           [](TreeIndex& self) { return self.total_node_nums(); })
      .def("get_nodes_given_level",
           [](TreeIndex& self, int level, bool ret_code) {
             return self.get_nodes_given_level(level, ret_code);
           })
      .def("get_parent_path",
           [](TreeIndex& self, std::vector<uint64_t>& ids, int start_level,
              bool ret_code) {
             return self.get_parent_path(ids, start_level, ret_code);
           })
      .def("get_ancestor_given_level",
           [](TreeIndex& self, const std::vector<uint64_t>& ids, int level,
              bool ret_code) {
             return self.get_ancestor_given_level(ids, level, ret_code);
           })
      .def("get_all_items",
           [](TreeIndex& self) { return self.get_all_items(); })
      .def("get_pi_relation",
           [](TreeIndex& self, const std::vector<uint64_t>& ids, int level) {
             return self.get_relation(level, ids);
           })
      .def("get_children_given_ancestor_and_level",
           [](TreeIndex& self, uint64_t ancestor, int level, bool ret_code) {
             return self.get_children_given_ancestor_and_level(ancestor, level,
                                                               ret_code);
           })
      .def("get_travel_path",
           [](TreeIndex& self, uint64_t child, uint64_t ancestor) {
             return self.get_travel_path(child, ancestor);
           })
      .def("tree_max_node",
           [](TreeIndex& self) { return self.tree_max_node(); });
}

void BindIndexWrapper(py::module* m) {
  py::class_<IndexWrapper, std::shared_ptr<IndexWrapper>>(*m, "IndexWrapper")
      .def(py::init([]() { return IndexWrapper::GetInstancePtr(); }))
      .def("insert_tree_index", &IndexWrapper::insert_tree_index)
      .def("get_tree_index", &IndexWrapper::GetTreeIndex)
      .def("clear_tree", &IndexWrapper::clear_tree);
}

using paddle::distributed::IndexSampler;
using paddle::distributed::LayerWiseSampler;
// using paddle::distributed::BeamSearchSampler;

void BindIndexSampler(py::module* m) {
  py::class_<IndexSampler, std::shared_ptr<IndexSampler>>(*m, "IndexSampler")
      .def(py::init([](const std::string& mode, const std::string& name) {
        if (mode == "by_layerwise") {
          return IndexSampler::Init<LayerWiseSampler>(name);
        } else {
          PADDLE_THROW(platform::errors::InvalidArgument(
              "Unsupported IndexSampler Type!"));
        }
      }))
      .def("init_layerwise_conf", &IndexSampler::init_layerwise_conf)
      .def("init_beamsearch_conf", &IndexSampler::init_beamsearch_conf)
      .def("sample", &IndexSampler::sample);
}

}  // end namespace pybind
}  // namespace paddle
