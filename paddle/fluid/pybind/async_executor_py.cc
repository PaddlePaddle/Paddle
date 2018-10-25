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

// To avoid conflicting definition in gcc-4.8.2 headers and pyconfig.h (2.7.3)
#ifdef _POSIX_C_SOURCE
#undef _POSIX_C_SOURCE
#endif

#ifdef _XOPEN_SOURCE
#undef _XOPEN_SOURCE
#endif

#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/variant.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/framework/async_executor_param.pb.h"
#include "paddle/fluid/framework/async_executor.h"
#include "paddle/fluid/pybind/async_executor_py.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {
void BindAsyncExecutor(py::module* m) {
  py::class_<paddle::AsyncExecutorParameter>(*m, "AsyncExecutorParameter")
    .def(py::init<>())
    .def("parse",
      [](paddle::AsyncExecutorParameter &self, const std::string &conf_file) {
        int file_descriptor = open(conf_file.c_str(), O_RDONLY);
        google::protobuf::io::FileInputStream file_input(file_descriptor);
        google::protobuf::TextFormat::Parse(&file_input, &self);
        close(file_descriptor);
      }
    );
  py::class_<framework::AsyncExecutor>(*m, "AsyncExecutor")
    .def(py::init<const platform::Place&>())
    .def("init",
      [](framework::AsyncExecutor &self,
          paddle::AsyncExecutorParameter &parameter,
          framework::Scope *scope) {
        paddle::BaseParameter base_param = parameter.base_param();

        // TODO Extract parameter list from python side, instead of
        // providing them in confgurations manually
        std::vector<std::string> param_names;
        for (int i = 0; i < base_param.model_param_names_size(); ++i) {
          param_names.push_back(base_param.model_param_names(i));
        }
#ifdef FORK_V1
        paddle::framework::InitDevices();
#else
        paddle::framework::InitDevices(false);
#endif
        self.InitRootScope(scope);
        self.SetThreadNum(base_param.thread_num());
        self.SetMaxTrainingEpoch(base_param.max_epoch());
        self.SetFileList(base_param.filelist().c_str());
        self.SetBatchSize(base_param.batch_size());
        self.SetDataFeedName(base_param.datafeed_class().c_str());
        self.SetInspectVarName(base_param.inspect_var_name());
        self.SetParamNames(param_names);
        self.SetModelPath(base_param.model_path());
        self.SetModelPrefix(base_param.model_prefix());
        self.SetInitProgFile(base_param.init_prog_file());
        self.SetInitModelFile(base_param.init_model_file());
        return;
      }
    )
    .def("run_startup_program", &framework::AsyncExecutor::RunStartupProgram)
    .def("load_init_model", &framework::AsyncExecutor::LoadInitModel)
    .def("run", &framework::AsyncExecutor::RunAsyncExecutor);
}   // end BindAsyncExecutor
}   // end namespace framework
}   // end namespace paddle

/* vim: set expandtab ts=2 sw=2 sts=2 tw=80: */
