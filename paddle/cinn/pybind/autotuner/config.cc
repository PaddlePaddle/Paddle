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

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/cast.h>
#include <vector>

#include "paddle/cinn/ir/group_schedule/config/group_tile_config.h"
#include "paddle/cinn/ir/group_schedule/config/database.h"
#include "paddle/cinn/ir/group_schedule/config/file_database.h"
#include "paddle/cinn/ir/group_schedule/config/schedule_config_manager.h"
#include "paddle/cinn/ir/group_schedule/search/config_searcher.h"
#include "paddle/cinn/ir/ir.h"


PD_DECLARE_string(tile_config_policy);
PD_DECLARE_string(cinn_tile_config_filename_label);
// COMMON_DECLARE_bool(print_ir);
PD_DECLARE_bool(cinn_measure_kernel_time);
PHI_DECLARE_bool(enable_cinn_compile_cache);

namespace cinn::pybind {

namespace py = pybind11;
using namespace cinn::ir;  // NOLINT



void BindTunerConfig(pybind11::module *m) {
    // 模块文档
    m->doc() = "Tuner configs and info";

    py::class_<NoneReduceMethod>(*m, "NoneReduceMethod")
        .def(py::init<>());
    py::class_<WarpReduceMethod>(*m, "WarpReduceMethod")
        .def(py::init<>());
    py::class_<BlockReduceMethod>(*m, "BlockReduceMethod")
        .def(py::init<>());
    py::class_<DiscreteReduceMethod>(*m, "DiscreteReduceMethod")
        .def(py::init<>());

    py::class_<ScheduleConfig::TileConfig>(*m, "TileConfig")
        .def(py::init<>())  // 默认构造函数
        .def(py::init<int, int, int, int, int, ReduceMethod>(),  // 带参数的构造函数
             py::arg("warp_num"),
             py::arg("tree_reduce_num"),
             py::arg("grid_reduce_num"),
             py::arg("spatial_inner_num"),
             py::arg("vectorize_factor"),
             py::arg("reduce_method"))
        .def_readwrite("warp_num", &ScheduleConfig::TileConfig::warp_num)
        .def_readwrite("tree_reduce_num", &ScheduleConfig::TileConfig::tree_reduce_num)
        .def_readwrite("grid_reduce_num", &ScheduleConfig::TileConfig::grid_reduce_num)
        .def_readwrite("spatial_inner_num", &ScheduleConfig::TileConfig::spatial_inner_num)
        .def_readwrite("vectorize_factor", &ScheduleConfig::TileConfig::vectorize_factor)
        .def_readwrite("reduce_method", &ScheduleConfig::TileConfig::reduce_method);


    // 首先绑定内部的Dimension结构
    py::class_<BucketInfo::Dimension>(*m, "Dimension")
        .def(py::init<>())  // 默认构造函数
        .def(py::init<int, int, std::string, bool>(),  // 带参数的构造函数
             py::arg("lower_bound"),
             py::arg("upper_bound"),
             py::arg("iter_type"),
             py::arg("is_dynamic"))
        .def_readwrite("lower_bound", &BucketInfo::Dimension::lower_bound)
        .def_readwrite("upper_bound", &BucketInfo::Dimension::upper_bound)
        .def_readwrite("iter_type", &BucketInfo::Dimension::iter_type)
        .def_readwrite("is_dynamic", &BucketInfo::Dimension::is_dynamic);

    // 绑定BucketInfo类
    py::class_<BucketInfo>(*m, "BucketInfo")
        // 构造函数
        .def(py::init<>())  // 默认构造函数
        .def(py::init<int, int, int, int, bool, bool>(),  // 六参数构造函数
             py::arg("sp_lower_bound"),
             py::arg("sp_upper_bound"),
             py::arg("rb_lower_bound"),
             py::arg("rb_upper_bound"),
             py::arg("sp_is_dynamic"),
             py::arg("rb_is_dynamic"))
        .def(py::init<size_t>(),  // size构造函数
             py::arg("size"))
        .def(py::init<const std::vector<BucketInfo::Dimension>&>(),  // vector构造函数
             py::arg("dims"))
        
        // 属性
        .def_readwrite("space", &BucketInfo::space)
        .def_readwrite("bucket_priority", &BucketInfo::bucket_priority)
        
        // 方法
        .def("__str__", &BucketInfo::ToString)
        .def("__eq__", &BucketInfo::operator==)
        
        // 常量
        .def_readonly_static("kMaxNumel", &BucketInfo::kMaxNumel)
        .def("__hash__", [](const BucketInfo& self) {
            return BucketInfoHash()(self);
        });
    // BucketInfo Hash
    // py::bind_map<TileConfigMap>(*m, "TileConfigMap");
    
    // // 绑定 IterSpaceType
    // py::bind_vector<IterSpaceType>(*m, "IterSpaceType")
    //     .def(py::init<>())
    //     .def("clear", &IterSpaceType::clear)
    //     .def("append", [](IterSpaceType& v, const std::pair<std::string, std::string>& p) {
    //         v.push_back(p);
    //     });

    // // 绑定 TileConfigMap
    // py::bind_map<TileConfigMap>(*m, "TileConfigMap")
    //     .def(py::init<>());

    // 绑定派生类 NaiveTileConfigDatabase
    py::class_<NaiveTileConfigDatabase>(*m, "NaiveTileConfigDatabase")
        .def(py::init<>())
        .def("add_config", 
            [](NaiveTileConfigDatabase& self,
            // const common::Target& target,
            const BucketInfo& bucket_info,
            const ScheduleConfig::TileConfig& config,
            int priority = 1) {
                self.AddConfig(cinn::common::DefaultTarget(), bucket_info, config, priority);
            },
            // py::arg("target"),
            py::arg("bucket_info"),
            py::arg("config"),
            py::arg("priority") = 1);
        // .def("get_configs", &NaiveTileConfigDatabase::GetConfigs);


        py::class_<ScheduleConfigManager, std::unique_ptr<ScheduleConfigManager, py::nodelete >>(*m, "ScheduleConfigManager")
            // 由于是单例模式，不需要构造函数绑定
            // 但需要提供获取实例的静态方法
            .def(py::init([](){ 
                return std::unique_ptr<ScheduleConfigManager, py::nodelete>(&ScheduleConfigManager::Instance());
            }))
            // .def_static("instance", &ScheduleConfigManager::Instance,
            //             py::return_value_policy::reference)
            
            // 成员函数绑定
            .def("add_config_database", 
                    &ScheduleConfigManager::AddConfigDatabase,
                    py::arg("id"),
                    py::arg("database"),
                    "Add a tile config database with specified ID")
            
            // .def("extract_configs",
            //         &ScheduleConfigManager::ExtractConfigs,
            //         py::arg("target"),
            //         py::arg("group_info"),
            //         "Extract configs for given target and fusion group")
            
            .def("set_policy",
                    &ScheduleConfigManager::SetPolicy,
                    py::arg("policy"),
                    "Set the schedule policy");

    py::class_<FileTileConfigDatabase>(*m, "FileTileConfigDatabase")
        .def(py::init<>())
        .def("add_config", 
            &FileTileConfigDatabase::AddConfig,
            py::arg("target"),
            py::arg("bucket_info"),
            py::arg("config"),
            py::arg("priority"))
        .def("get_configs", &FileTileConfigDatabase::GetConfigs);
    
        

    // 绑定函数
    m->def("_tuner_add_config_helper", 
        &cinn::ir::search::TunerAddConfigHelper,
        py::arg("candidate"),
        py::arg("bucket_info"),
        R"pbdoc(
          Add configuration to the tile config database based on candidate and bucket info.
          
          Args:
              candidate: A list of integers representing the candidate configuration
              bucket_info: BucketInfo object containing space information
          
          The function creates a new NaiveTileConfigDatabase and configures it with:
              - warp_num from candidate[0]
              - tree_reduce_num from candidate[1]
              - spatial_inner_num from candidate[2]
        )pbdoc");
    
    // py::module env_controller = 
    //     m->def_submodule("EnvController", "Compiler environment variable controller");
    // env_controller.def

    m->def("_env_set_tile_config_policy",
        [&](const std::string& policy){ FLAGS_tile_config_policy = policy;},
        py::arg("policy")
    );
    m->def("_env_get_tile_config_policy",
        [&](){ return FLAGS_tile_config_policy;}
    );
}

}
