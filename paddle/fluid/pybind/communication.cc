/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <chrono>
#include <string>

#include "paddle/fluid/distributed/store/tcp_store.h"
#include "paddle/fluid/pybind/communication.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

using TCPStore = paddle::distributed::TCPStore;

void BindTCPStore(py::module *m) {
  auto Store =
      py::class_<distributed::Store, std::shared_ptr<distributed::Store>>(
          *m, "Store")
          .def(py::init<>())
          .def("set",
               [](distributed::Store &self, const std::string &key,
                  const std::string &value) {
                 std::vector<uint8_t> data(value.begin(), value.end());
                 self.set(key, data);
               },
               py::arg("key"), py::arg("value"),
               py::call_guard<py::gil_scoped_release>(), R"DOC(
               The OP inserts the input key-value into the store, and if the key already 
               exists in the store, it overwrites the old value with the newly supplied 
               value.

               Args:
                    key (str): The given key in the store.
                    value (str): The value associated with the key added to the store.

               Returns:
                    None

               Examples:
               .. code-block:: python
                    import datetime
                    import paddle

                    store = paddle.fluid.core.TCPStore("127.0.0.1", 6170, True, 1,
                                                            datetime.timedelta(0))

                    store.set("key",3)
                    ret = store.get("key")
                    print(ret)
               )DOC")
          .def("get",
               [](distributed::Store &self,
                  const std::string &key) -> py::bytes {
                 auto data = self.get(key);
                 return py::bytes(reinterpret_cast<char *>(data.data()),
                                  data.size());
               },
               py::arg("key"), py::call_guard<py::gil_scoped_release>(), R"DOC(
               The OP returns the value of the given key.

               Args:
                    key (str): The given key in the store.

               Returns:
                    Returns the value of the given key in the store.

               Examples:
               .. code-block:: python
                    import datetime
                    import paddle

                    store = paddle.fluid.core.TCPStore("127.0.0.1", 6170, True, 1,
                                                            datetime.timedelta(0))
                    store.add("my", 3)
                    ret = store.get('my')
                    print(ret)
               )DOC")
          .def("add", &distributed::Store::add,
               py::call_guard<py::gil_scoped_release>(), R"DOC(
               The OP creates a counter associated with the key and initializes it
               as value on its first call. Then it calls to use the same key counter
               to increase the amount.

               Args:
                    key (str):The key that increments the counter in the store.
                    value(int):The value increased by the counter.

               Returns:
                    None.

               Examples:
               .. code-block:: python
                    import datetime
                    import paddle

                    store = paddle.fluid.core.TCPStore("127.0.0.1", 6170, True, 1,
                                                            datetime.timedelta(0))
                    store.add("my", 3)
                    store.add("my", 3)
                    ret = store.get('my')
                    print(ret)
               )DOC")
          .def("wait", &distributed::Store::wait,
               py::call_guard<py::gil_scoped_release>(), R"DOC(
               The OP throws an exception for adding a key to a storage timeout.

               Args:
                    key (str): The key that needs to wait.

               Returns:
                    None.

               Examples:
               .. code-block:: python

               import datetime
               import paddle

               store = paddle.fluid.core.TCPStore("127.0.0.1", 6170, True, 1,
                                                       datetime.timedelta(0))
               store.wait("my")

               )DOC");

  py::class_<TCPStore, std::shared_ptr<TCPStore>>(*m, "TCPStore", Store)
      .def(py::init([](std::string hostname, uint16_t port, bool is_master,
                       size_t world_size, std::chrono::seconds timeout) {
             return std::make_shared<TCPStore>(hostname, port, is_master,
                                               world_size, timeout);
           }),
           py::arg("hostname"), py::arg("port"), py::arg("is_master"),
           py::arg("world_size"),
           py::arg("timeout") = distributed::tcputils::kNoTimeout,
           py::call_guard<py::gil_scoped_release>(), R"DOC(
          The OP is to realize the storage of TCP distributed key-value. By the 
          initialization of the server storage, the client will connect to the 
          server through TCP, which can insert key value pair using set(), return 
          key value peer using get() and some operations.

          Args:
               host_name (str): The host name or IP address of the server.
               port (int): Port on which the server listens for requests.
               world_size (int, optional): Total number of servers and clients in the 
                                           store. The default value is 1.
               is_master(bool, optional): True initializes the server and False initializes 
                                          the client. The default value is False.
               timeout(timedelta, optional): Maximum timeout allowed for initialization storage.
                                             The default value is 360s.

          Returns:
               None.

          Examples:
            .. code-block:: python
               import datetime
               import paddle

               store = paddle.fluid.core.TCPStore("127.0.0.1", 6170, True, 1,
                                                       datetime.timedelta(0))
               store.add("my", 3)
               ret = store.get('my')
               print(ret)
          )DOC");
}

}  // namespace pybind
}  // namespace paddle
