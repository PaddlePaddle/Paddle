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
#include <memory>
#include <string>
#include <vector>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "paddle/fluid/framework/async_executor.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/data_feed.pb.h"
#include "paddle/fluid/framework/data_set.h"
#include "paddle/fluid/framework/dataset_factory.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/variant.h"
#include "paddle/fluid/pybind/data_set_py.h"

namespace py = pybind11;
namespace pd = paddle::framework;

namespace paddle {
namespace pybind {

void BindDataset(py::module* m) {
  py::class_<framework::Dataset, std::unique_ptr<framework::Dataset>>(*m,
                                                                      "Dataset")
      .def(py::init([](const std::string& name = "MultiSlotDataset") {
        return framework::DatasetFactory::CreateDataset(name);
      }))
      .def("set_filelist", &framework::Dataset::SetFileList,
           py::call_guard<py::gil_scoped_release>())
      .def("set_thread_num", &framework::Dataset::SetThreadNum,
           py::call_guard<py::gil_scoped_release>())
      .def("set_trainer_num", &framework::Dataset::SetTrainerNum,
           py::call_guard<py::gil_scoped_release>())
      .def("set_fleet_send_batch_size",
           &framework::Dataset::SetFleetSendBatchSize,
           py::call_guard<py::gil_scoped_release>())
      .def("set_hdfs_config", &framework::Dataset::SetHdfsConfig,
           py::call_guard<py::gil_scoped_release>())
      .def("set_data_feed_desc", &framework::Dataset::SetDataFeedDesc,
           py::call_guard<py::gil_scoped_release>())
      .def("get_filelist", &framework::Dataset::GetFileList,
           py::call_guard<py::gil_scoped_release>())
      .def("get_thread_num", &framework::Dataset::GetThreadNum,
           py::call_guard<py::gil_scoped_release>())
      .def("get_trainer_num", &framework::Dataset::GetTrainerNum,
           py::call_guard<py::gil_scoped_release>())
      .def("get_fleet_send_batch_size",
           &framework::Dataset::GetFleetSendBatchSize,
           py::call_guard<py::gil_scoped_release>())
      .def("get_hdfs_config", &framework::Dataset::GetHdfsConfig,
           py::call_guard<py::gil_scoped_release>())
      .def("get_data_feed_desc", &framework::Dataset::GetDataFeedDesc,
           py::call_guard<py::gil_scoped_release>())
      .def("register_client2client_msg_handler",
           &framework::Dataset::RegisterClientToClientMsgHandler,
           py::call_guard<py::gil_scoped_release>())
      .def("create_channel", &framework::Dataset::CreateChannel,
           py::call_guard<py::gil_scoped_release>())
      .def("create_readers", &framework::Dataset::CreateReaders,
           py::call_guard<py::gil_scoped_release>())
      .def("destroy_readers", &framework::Dataset::DestroyReaders,
           py::call_guard<py::gil_scoped_release>())
      .def("load_into_memory", &framework::Dataset::LoadIntoMemory,
           py::call_guard<py::gil_scoped_release>())
      .def("preload_into_memory", &framework::Dataset::PreLoadIntoMemory,
           py::call_guard<py::gil_scoped_release>())
      .def("wait_preload_done", &framework::Dataset::WaitPreLoadDone,
           py::call_guard<py::gil_scoped_release>())
      .def("release_memory", &framework::Dataset::ReleaseMemory,
           py::call_guard<py::gil_scoped_release>())
      .def("local_shuffle", &framework::Dataset::LocalShuffle,
           py::call_guard<py::gil_scoped_release>())
      .def("global_shuffle", &framework::Dataset::GlobalShuffle,
           py::call_guard<py::gil_scoped_release>())
      .def("get_memory_data_size", &framework::Dataset::GetMemoryDataSize,
           py::call_guard<py::gil_scoped_release>())
      .def("get_shuffle_data_size", &framework::Dataset::GetShuffleDataSize,
           py::call_guard<py::gil_scoped_release>())
      .def("set_queue_num", &framework::Dataset::SetChannelNum,
           py::call_guard<py::gil_scoped_release>());
}

}  // end namespace pybind
}  // end namespace paddle
