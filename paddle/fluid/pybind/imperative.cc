/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/pybind/imperative.h"

#include <Python.h>
#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "paddle/fluid/imperative/backward_strategy.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/imperative/nccl_context.h"
#include "paddle/fluid/imperative/profiler.h"
#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/fluid/pybind/pybind_boost_headers.h"

namespace paddle {
namespace pybind {

namespace py = ::pybind11;

template <typename P>
extern void SetTensorFromPyArray(framework::Tensor *self, pybind11::array array,
                                 P place);
extern py::array TensorToPyArray(const framework::Tensor &tensor,
                                 bool need_deep_copy = false);

class Layer : public imperative::Layer {
 public:
  using imperative::Layer::Layer;  // Inherit constructors

  std::vector<std::shared_ptr<imperative::VarBase>> Forward(
      const std::vector<std::shared_ptr<imperative::VarBase>> &inputs)
      override {
    PYBIND11_OVERLOAD(std::vector<std::shared_ptr<imperative::VarBase>>, Layer,
                      Forward, inputs);  // NOLINT
  }
};

template <typename P>
static void InitVarBaseFromNumpy(imperative::VarBase *self,
                                 const std::string &name, bool persistable,
                                 const py::array &array, const P &place) {
  new (self) imperative::VarBase(name);
  self->SetPersistable(persistable);
  auto *tensor = self->MutableVar()->GetMutable<framework::LoDTensor>();
  SetTensorFromPyArray<P>(tensor, array, place);
  self->SetType(framework::proto::VarType::LOD_TENSOR);
  self->SetDataType(tensor->type());
}

static std::string GetTypeName(const imperative::VarBase &var) {
  if (var.Type() == framework::proto::VarType::RAW) {
    return "RAW";
  } else if (!var.Var().IsInitialized()) {
    return "nullptr";
  } else {
    return framework::ToTypeName(var.Var().Type());
  }
}

// Bind Methods
void BindImperative(py::module *m_ptr) {
  auto &m = *m_ptr;

  py::class_<imperative::detail::BackwardStrategy> backward_strategy(
      m, "BackwardStrategy", R"DOC(

    BackwardStrategy is a descriptor of how to run the backward process.

    **Note**:
        **This API is only avaliable in** `Dygraph <../../user_guides/howto/dygraph/DyGraph.html>`_ **Mode**

    Attribute:
        **sort_sum_gradient**:

        If framework will sum the gradient by the reverse order of trace. eg. x_var ( :ref:`api_guide_Variable` ) will be the input of multiple OP such as :ref:`api_fluid_layers_scale` , this attr will decide if framework will sum gradient of `x_var` by the reverse order.

        By Default: False

        Examples:
            .. code-block:: python

                import numpy as np
                import paddle.fluid as fluid

                x = np.ones([2, 2], np.float32)
                with fluid.dygraph.guard():
                    x_var = fluid.dygraph.to_variable(x)
                    sums_inputs = []
                    # x_var will be multi-scales' input here
                    for _ in range(10):
                        sums_inputs.append(fluid.layers.scale(x_var))
                    ret2 = fluid.layers.sums(sums_inputs)
                    loss2 = fluid.layers.reduce_sum(ret2)
                    backward_strategy = fluid.dygraph.BackwardStrategy()
                    backward_strategy.sort_sum_gradient = True
                    loss2.backward(backward_strategy)
      )DOC");
  backward_strategy.def(py::init())
      .def_property("sort_sum_gradient",
                    [](const imperative::detail::BackwardStrategy &self) {
                      return self.sorted_sum_gradient_;
                    },
                    [](imperative::detail::BackwardStrategy &self,
                       bool sorted_sum_gradient) {
                      self.sorted_sum_gradient_ = sorted_sum_gradient;
                    });

  m.def("start_imperative_gperf_profiler",
        []() { imperative::StartProfile(); });

  m.def("stop_imperative_gperf_profiler", []() { imperative::StopProfile(); });

  m.def("_is_dygraph_debug_enabled",
        []() { return imperative::IsDebugEnabled(); });
  m.def("_dygraph_debug_level", []() { return imperative::GetDebugLevel(); });

  py::class_<imperative::VarBase, std::shared_ptr<imperative::VarBase>>(
      m, "VarBase",
      R"DOC()DOC")
      .def_static("_alive_vars", &imperative::VarBase::AliveVarNames)
      .def("__init__",
           [](imperative::VarBase &self, framework::proto::VarType::Type dtype,
              const std::vector<int> &dims, const std::string &name,
              framework::proto::VarType::Type type, bool persistable) {
             new (&self) imperative::VarBase(name);
             self.SetPersistable(persistable);
             self.SetType(type);
             self.SetDataType(dtype);
             if (type == framework::proto::VarType::LOD_TENSOR) {
               auto *tensor =
                   self.MutableVar()->GetMutable<framework::LoDTensor>();
               tensor->Resize(framework::make_ddim(dims));
             }
           })
      .def("__init__", InitVarBaseFromNumpy<platform::CPUPlace>,
           py::arg("name"), py::arg("persistable"), py::arg("value"),
           py::arg("place"))
      .def("__init__", InitVarBaseFromNumpy<platform::CUDAPlace>,
           py::arg("name"), py::arg("persistable"), py::arg("value"),
           py::arg("place"))
      .def("__init__", InitVarBaseFromNumpy<platform::CUDAPinnedPlace>,
           py::arg("name"), py::arg("persistable"), py::arg("value"),
           py::arg("place"))
      .def("numpy",
           [](imperative::VarBase &self) -> py::array {
             const auto &tensor =
                 self.MutableVar()->Get<framework::LoDTensor>();
             PADDLE_ENFORCE_EQ(tensor.IsInitialized(), true,
                               "%s is Empty, Please check if it has no data in",
                               self.Name());
             return TensorToPyArray(tensor, true);
           },
           R"DOC(
        **Notes**:
            **This API is ONLY avaliable in Dygraph mode**

        Returns a numpy array shows the value of current :ref:`api_guide_Variable_en`

        Returns:
            ndarray: The numpy value of current Variable.

        Returns type:
            ndarray: dtype is same as current Variable

        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                from paddle.fluid.dygraph.base import to_variable
                from paddle.fluid.dygraph import FC
                import numpy as np

                data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
                with fluid.dygraph.guard():
                    fc = FC("fc", 64, num_flatten_dims=2)
                    data = to_variable(data)
                    x = fc(data)
                    print(x.numpy())

       )DOC")
      .def("detach",
           [](const imperative::VarBase &self) {
             const auto &tensor = self.Var().Get<framework::LoDTensor>();
             PADDLE_ENFORCE_EQ(tensor.IsInitialized(), true,
                               "%s has not been initialized", self.Name());
             return self.NewVarBase(tensor.place(), false);
           },
           py::return_value_policy::copy, R"DOC(
        **Notes**:
            **This API is ONLY avaliable in Dygraph mode**

        Returns a new Variable, detached from the current graph.

        Returns:
             ( :ref:`api_guide_Variable_en` | dtype is same as current Variable): The detached Variable.


        Examples:
            .. code-block:: python

                import paddle.fluid as fluid
                from paddle.fluid.dygraph.base import to_variable
                from paddle.fluid.dygraph import FC
                import numpy as np

                data = np.random.uniform(-1, 1, [30, 10, 32]).astype('float32')
                with fluid.dygraph.guard():
                    fc = FC("fc", 64, num_flatten_dims=2)
                    data = to_variable(data)
                    x = fc(data)
                    y = x.detach()

       )DOC")
      .def("_run_backward",
           [](imperative::VarBase &self,
              const imperative::detail::BackwardStrategy &bckst,
              const imperative::Tracer &tracer) {
             // TODO(jiabin): when we impl more backward execution we can select
             // them

             imperative::Engine *engine = tracer.GetDefaultEngine();
             VLOG(3) << "Start backward";
             engine->Init(&self, bckst);
             engine->Execute();
             VLOG(3) << "Finish backward";
           },
           py::call_guard<py::gil_scoped_release>())
      .def("_grad_name", &imperative::VarBase::GradVarName)
      .def("_grad_value",
           [](imperative::VarBase &self) {
             return self.MutableGradVar()->Get<framework::LoDTensor>();
           },
           py::return_value_policy::reference)
      .def("clear_gradient", &imperative::VarBase::ClearGradient, R"DOC(

        **Notes**:
        **1. This API is ONLY avaliable in Dygraph mode**

        **2. Use it only Variable has gradient, normally we use this for Parameters since other temporal Variable will be deleted by Python's GC**

        Clear  (set to ``0`` ) the Gradient of Current Variable

        Returns:  None

        Examples:
             .. code-block:: python

                import paddle.fluid as fluid
                import numpy as np

                x = np.ones([2, 2], np.float32)
                with fluid.dygraph.guard():
                    inputs2 = []
                    for _ in range(10):
                         tmp = fluid.dygraph.base.to_variable(x)
                         tmp.stop_gradient=False
                         inputs2.append(tmp)
                    ret2 = fluid.layers.sums(inputs2)
                    loss2 = fluid.layers.reduce_sum(ret2)
                    backward_strategy = fluid.dygraph.BackwardStrategy()
                    backward_strategy.sort_sum_gradient = True
                    loss2.backward(backward_strategy)
                    print(loss2.gradient())
                    loss2.clear_gradient()
                    print("After clear {}".format(loss2.gradient()))
      )DOC")
      .def("_grad_ivar",
           [](const imperative::VarBase &self) {
             auto &grad_var = self.GradVarBase();
             if (grad_var && grad_var->Var().IsInitialized()) {
               return grad_var;
             } else {
               return std::shared_ptr<imperative::VarBase>(nullptr);
             }
           },
           py::return_value_policy::copy)
      .def("_copy_to",
           [](const imperative::VarBase &self, const platform::CPUPlace &place,
              bool blocking) { return self.NewVarBase(place, blocking); },
           py::return_value_policy::copy)
      .def("_copy_to",
           [](const imperative::VarBase &self, const platform::CUDAPlace &place,
              bool blocking) { return self.NewVarBase(place, blocking); },
           py::return_value_policy::copy)
      .def("value", [](imperative::VarBase &self) { return self.MutableVar(); },
           py::return_value_policy::reference)
      .def_property("name", &imperative::VarBase::Name,
                    &imperative::VarBase::SetName)
      .def_property_readonly(
          "shape",
          [](imperative::VarBase &self) {
            if (self.Var().IsType<framework::LoDTensor>()) {
              return framework::vectorize<int>(
                  self.Var().Get<framework::LoDTensor>().dims());
            } else {
              VLOG(2) << "It is meaningless to get shape of variable type "
                      << GetTypeName(self);
              return std::vector<int>();
            }
          })
      .def_property_readonly("type", &imperative::VarBase::Type)
      .def_property_readonly("dtype", &imperative::VarBase::DataType)
      .def_property("persistable", &imperative::VarBase::Persistable,
                    &imperative::VarBase::SetPersistable)
      .def_property("stop_gradient",
                    &imperative::VarBase::OverridedStopGradient,
                    &imperative::VarBase::SetOverridedStopGradient);

  py::class_<imperative::Layer, Layer /* <--- trampoline*/> layer(m, "Layer");
  layer.def(py::init<>())
      .def("forward",
           [](imperative::Layer &self,
              const std::vector<std::shared_ptr<imperative::VarBase>> &inputs) {
             return self.Forward(inputs);
           });

  py::class_<imperative::jit::ProgramDescTracer>(m, "ProgramDescTracer", "")
      .def("set_name_prefix",
           &imperative::jit::ProgramDescTracer::SetNamePrefix)
      .def("set_feed_vars", &imperative::jit::ProgramDescTracer::SetFeedVars)
      .def("set_fetch_vars", &imperative::jit::ProgramDescTracer::SetFetchVars)
      .def("create_program_desc",
           &imperative::jit::ProgramDescTracer::CreateProgramDesc)
      .def("reset", &imperative::jit::ProgramDescTracer::Reset);

  py::class_<imperative::Tracer>(m, "Tracer", "")
      .def("__init__",
           [](imperative::Tracer &self) { new (&self) imperative::Tracer(); })
      .def_property("_enable_program_desc_tracing",
                    &imperative::Tracer::IsProgramDescTracingEnabled,
                    &imperative::Tracer::SetEnableProgramDescTracing)
      .def("_get_program_desc_tracer",
           &imperative::Tracer::GetProgramDescTracer,
           py::return_value_policy::reference)
      .def("trace",
           [](imperative::Tracer &self, const std::string &type,
              const imperative::NameVarBaseMap &ins,
              const imperative::NameVarBaseMap &outs,
              framework::AttributeMap attrs, const platform::CUDAPlace &place,
              bool trace_backward) {
             {
               py::gil_scoped_release release;
               self.TraceOp(type, std::move(ins), std::move(outs),
                            std::move(attrs), place, trace_backward);
             }
           })
      .def("trace", [](imperative::Tracer &self, const std::string &type,
                       const imperative::NameVarBaseMap &ins,
                       const imperative::NameVarBaseMap &outs,
                       framework::AttributeMap attrs,
                       const platform::CPUPlace &place, bool trace_backward) {
        {
          py::gil_scoped_release release;
          self.TraceOp(type, std::move(ins), std::move(outs), std::move(attrs),
                       place, trace_backward);
        }
      });

  // define parallel context
  py::class_<imperative::ParallelStrategy> parallel_strategy(
      m, "ParallelStrategy", "");
  parallel_strategy.def(py::init())
      .def_property(
          "nranks",
          [](const imperative::ParallelStrategy &self) { return self.nranks_; },
          [](imperative::ParallelStrategy &self, int nranks) {
            self.nranks_ = nranks;
          })
      .def_property("local_rank",
                    [](const imperative::ParallelStrategy &self) {
                      return self.local_rank_;
                    },
                    [](imperative::ParallelStrategy &self, int local_rank) {
                      self.local_rank_ = local_rank;
                    })
      .def_property(
          "trainer_endpoints",
          [](const imperative::ParallelStrategy &self) {
            return self.trainer_endpoints_;
          },
          [](imperative::ParallelStrategy &self, std::vector<std::string> eps) {
            self.trainer_endpoints_ = eps;
          })
      .def_property("current_endpoint",
                    [](const imperative::ParallelStrategy &self) {
                      return self.current_endpoint_;
                    },
                    [](imperative::ParallelStrategy &self,
                       const std::string &ep) { self.current_endpoint_ = ep; });
#if defined(PADDLE_WITH_CUDA) && !defined(_WIN32)
  py::class_<imperative::NCCLParallelContext> nccl_ctx(m,
                                                       "NCCLParallelContext");

  nccl_ctx
      .def(py::init<const imperative::ParallelStrategy &,
                    const platform::CUDAPlace &>())
      .def("init", [](imperative::NCCLParallelContext &self) { self.Init(); });
#endif
}

}  // namespace pybind
}  // namespace paddle
