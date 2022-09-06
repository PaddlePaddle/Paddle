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

#include "paddle/fluid/pybind/communication.h"

#include <Python.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include <chrono>
#include <string>

#include "paddle/fluid/distributed/store/tcp_store.h"

namespace py = pybind11;

namespace paddle {
namespace pybind {

using TCPStore = paddle::distributed::TCPStore;

void BindTCPStore(py::module *m) {
  auto Store =
      py::class_<distributed::Store, std::shared_ptr<distributed::Store>>(
          *m, "Store")
          .def(py::init<>())
          .def(
              "set",
              [](distributed::Store &self,
                 const std::string &key,
                 const std::string &value) {
                std::vector<uint8_t> data(value.begin(), value.end());
                self.set(key, data);
              },
              py::arg("key"),
              py::arg("value"),
              py::call_guard<py::gil_scoped_release>(),
              R"DOC(
      Insert the key-value pair into the store. If key already exists in the store, it will overwrite the old value with the new supplied value.

      Parameters:
        key(str): The key to be added to the store.
        value(str): The value associated with key to be added to the store.

      Examples:
        .. code-block:: python
        import paddle.distributed as dist

        store = dist.TCPStore("localhost", 0, 1, True)
        store.set("first_key", "first_value")
        store.get("first_key") # return "b'first_value"
)DOC")
          .def(
              "get",
              [](distributed::Store &self,
                 const std::string &key) -> py::bytes {
                auto data = self.get(key);
                return py::bytes(reinterpret_cast<char *>(data.data()),
                                 data.size());
              },
              py::arg("key"),
              py::call_guard<py::gil_scoped_release>(),
              R"DOC(
      Get the value associated with the given key in the store. If key is not present in the store, this method will wait for timeout before throwing an exception.

      Parameters:
        key(str): This method will return the value associated with this key.

      Return:
        byte_str: Value associated with key if key is in the store.

      Examples:
        .. code-block:: python
        import paddle.distributed as dist

        store = dist.TCPStore("localhost", 0, 1, True)
        store.set("first_key", "first_value")
        store.get("first_key") # return "b'first_value"
)DOC")
          .def("add",
               &distributed::Store::add,
               py::call_guard<py::gil_scoped_release>(),
               R"DOC(
      Find the value associated with the specifc key and increase it using the specified value. If the key not exists, initialized it using the specified value.

      Parameters:
        key(str): The key in the store which value will be incremented.
        value(int): The value to be increased.

      Return:
        byte_str: Value associated with key after increment.

      Examples:
        .. code-block:: python
        import paddle.distributed as dist

        store = dist.TCPStore("localhost", 0, 1, True)
        store.add("first_key", 1)
        store.add("first_key", 6)
        store.get("first_key") # return "b'7"
)DOC")
          .def("wait",
               &distributed::Store::wait,
               py::call_guard<py::gil_scoped_release>(),
               R"DOC(
      Wait for the key to be set in the store. This method will wait for timeout before throwing an exception.

      Parameters:
        key(str): This method will return the value associated with this key.

      Examples:
        .. code-block:: python
        import paddle.distributed as dist

        store = dist.TCPStore("localhost", 0, 1, True)
        store.set("first_key", "first_value")
        store.wait("first_key")
)DOC");

  py::class_<TCPStore, std::shared_ptr<TCPStore>>(*m, "TCPStore", Store, R"DOC(

    TCPStore is a key-value store implementation based on TCP-based. There is only one server which holds the data, and the clients connect to the server based on TCP. There are several actions such as get() to retrieve the value of specific key, set() to insert a new key-value pair, etc.

    Parameters:
        hostname (string): The hostname or direct IP address indicate the server store should run on.
        port (int): The listening port of server.
        is_master (bool): Indicate the store is server or client. True for server, false for client.
        world_size (int): The number of processes need to be connected.
        timeout (int, optional): Timeout used by the store during initialization and for methods such as get() and wait(). Default is 900 seconds.

    Examples:
        .. code-block:: python

          import paddle.distributed as dist
          rank = dist.get_rank()
          world_size = dist.get_world_size()
          is_master = (rank == 0)
          store = dist.TCPStore(hostname="localhost", port=12345, is_master=is_master, world_size=world_size)
          if is_master:
            store.get("first_key")
          else:
            store.set("first_key", "first_value")

)DOC")
      .def(py::init([](std::string hostname,
                       uint16_t port,
                       bool is_master,
                       size_t world_size,
                       int timeout) {
             return std::make_shared<TCPStore>(
                 hostname, port, is_master, world_size, timeout);
           }),
           py::arg("hostname"),
           py::arg("port"),
           py::arg("is_master"),
           py::arg("world_size"),
           py::arg("timeout") = 900,
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace pybind
}  // namespace paddle
