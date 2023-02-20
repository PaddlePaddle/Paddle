#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif
#include  "paddle/fluid/imperative/tracer.h"
#include  "paddle/fluid/platform/profiler.h"
#include  "pybind11/numpy.h"
#include  "pybind11/pybind11.h"
#include  "pybind11/detail/common.h"
#include  "paddle/fluid/pybind/eager_utils.h"
#include  "paddle/fluid/pybind/op_function.h"
#include  <Python.h>


namespace paddle {
namespace pybind {

extern std::atomic<int> VarBaseUniqueNameID;

static PyObject * imperative_reshape2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reshape2";
    platform::RecordEvent op_type_record_event("reshape2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Shape = GetVarBaseFromArgs(op_type, "Shape", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Shape != nullptr) {
      ins["Shape"] = {Shape};
    }

    if (ins.count("X") && outs.count("Out")) {
      HandleViewBetweenInputAndOutput(ins["X"][0], outs["Out"][0]);
    }
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reshape2_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reshape2";
    platform::RecordEvent op_type_record_event("reshape2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Shape = GetVarBaseFromArgs(op_type, "Shape", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Shape != nullptr) {
      ins["Shape"] = {Shape};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_reduce_any(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reduce_any";
    platform::RecordEvent op_type_record_event("reduce_any pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_limit_by_capacity(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "limit_by_capacity";
    platform::RecordEvent op_type_record_event("limit_by_capacity pybind_imperative_func");
    
    auto expert_count = GetVarBaseFromArgs(op_type, "expert_count", args, 0, false);
    auto capacity = GetVarBaseFromArgs(op_type, "capacity", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"expert_count", {expert_count}},{"capacity", {capacity}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unstack(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unstack";
    platform::RecordEvent op_type_record_event("unstack pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto YNum = GetUnsignedLongFromArgs(op_type, "YNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", ConstructDuplicableOutput(YNum)}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_scatter_nd_add(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "scatter_nd_add";
    platform::RecordEvent op_type_record_event("scatter_nd_add pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    auto Updates = GetVarBaseFromArgs(op_type, "Updates", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Index", {Index}},{"Updates", {Updates}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_reshape(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_reshape";
    platform::RecordEvent op_type_record_event("sequence_reshape pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bilateral_slice(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bilateral_slice";
    platform::RecordEvent op_type_record_event("bilateral_slice pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Grid = GetVarBaseFromArgs(op_type, "Grid", args, 1, false);
    auto Guide = GetVarBaseFromArgs(op_type, "Guide", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Grid", {Grid}},{"Guide", {Guide}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill_any_like(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_any_like";
    platform::RecordEvent op_type_record_event("fill_any_like pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_empty(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "empty";
    platform::RecordEvent op_type_record_event("empty pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pad_constant_like(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pad_constant_like";
    platform::RecordEvent op_type_record_event("pad_constant_like pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pool2d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pool2d";
    platform::RecordEvent op_type_record_event("pool2d pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_size(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "size";
    platform::RecordEvent op_type_record_event("size pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_imag(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "imag";
    platform::RecordEvent op_type_record_event("imag pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pull_gpups_sparse(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pull_gpups_sparse";
    platform::RecordEvent op_type_record_event("pull_gpups_sparse pybind_imperative_func");
    
    auto Ids = GetVarBaseListFromArgs(op_type, "Ids", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"Ids", Ids}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_eigh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "eigh";
    platform::RecordEvent op_type_record_event("eigh pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Eigenvalues", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Eigenvectors", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Eigenvalues"][0],outs["Eigenvectors"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_stack(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "stack";
    platform::RecordEvent op_type_record_event("stack pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dgc_momentum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dgc_momentum";
    platform::RecordEvent op_type_record_event("dgc_momentum pybind_imperative_func");
    
    auto current_step = GetVarBaseFromArgs(op_type, "current_step", args, 0, false);
    auto nranks = GetVarBaseFromArgs(op_type, "nranks", args, 1, false);
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 2, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 3, false);
    auto Velocity = GetVarBaseFromArgs(op_type, "Velocity", args, 4, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Grad_out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ParamOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"VelocityOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"current_step", {current_step}},{"nranks", {nranks}},{"Param", {Param}},{"Grad", {Grad}},{"Velocity", {Velocity}},{"LearningRate", {LearningRate}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Grad_out"][0],outs["ParamOut"][0],outs["VelocityOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lamb(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lamb";
    platform::RecordEvent op_type_record_event("lamb pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 2, false);
    auto Moment1 = GetVarBaseFromArgs(op_type, "Moment1", args, 3, false);
    auto Moment2 = GetVarBaseFromArgs(op_type, "Moment2", args, 4, false);
    auto Beta1Pow = GetVarBaseFromArgs(op_type, "Beta1Pow", args, 5, false);
    auto Beta2Pow = GetVarBaseFromArgs(op_type, "Beta2Pow", args, 6, false);
    auto MasterParam = GetVarBaseFromArgs(op_type, "MasterParam", args, 7, true);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 8, false);
    auto Moment1Out = GetVarBaseFromArgs(op_type, "Moment1Out", args, 9, false);
    auto Moment2Out = GetVarBaseFromArgs(op_type, "Moment2Out", args, 10, false);
    auto Beta1PowOut = GetVarBaseFromArgs(op_type, "Beta1PowOut", args, 11, true);
    auto Beta2PowOut = GetVarBaseFromArgs(op_type, "Beta2PowOut", args, 12, true);
    auto MasterParamOut = GetVarBaseFromArgs(op_type, "MasterParamOut", args, 13, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 14, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"Moment1Out", {Moment1Out}},{"Moment2Out", {Moment2Out}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"LearningRate", {LearningRate}},{"Moment1", {Moment1}},{"Moment2", {Moment2}},{"Beta1Pow", {Beta1Pow}},{"Beta2Pow", {Beta2Pow}}};
    
    if (MasterParam != nullptr) {
      ins["MasterParam"] = {MasterParam};
    }

    outs["Beta1PowOut"] = {Beta1PowOut};

    outs["Beta2PowOut"] = {Beta2PowOut};

    outs["MasterParamOut"] = {MasterParamOut};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["Moment1Out"][0],outs["Moment2Out"][0],outs["Beta1PowOut"][0],outs["Beta2PowOut"][0],outs["MasterParamOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_generate_proposals_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "generate_proposals_v2";
    platform::RecordEvent op_type_record_event("generate_proposals_v2 pybind_imperative_func");
    
    auto Scores = GetVarBaseFromArgs(op_type, "Scores", args, 0, false);
    auto BboxDeltas = GetVarBaseFromArgs(op_type, "BboxDeltas", args, 1, false);
    auto ImShape = GetVarBaseFromArgs(op_type, "ImShape", args, 2, false);
    auto Anchors = GetVarBaseFromArgs(op_type, "Anchors", args, 3, false);
    auto Variances = GetVarBaseFromArgs(op_type, "Variances", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"RpnRois", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RpnRoiProbs", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RpnRoisNum", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Scores", {Scores}},{"BboxDeltas", {BboxDeltas}},{"ImShape", {ImShape}},{"Anchors", {Anchors}},{"Variances", {Variances}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["RpnRois"][0],outs["RpnRoiProbs"][0],outs["RpnRoisNum"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bitwise_or(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bitwise_or";
    platform::RecordEvent op_type_record_event("bitwise_or pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_gru_unit(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gru_unit";
    platform::RecordEvent op_type_record_event("gru_unit pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto HiddenPrev = GetVarBaseFromArgs(op_type, "HiddenPrev", args, 1, false);
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 2, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Gate", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ResetHiddenPrev", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Hidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"HiddenPrev", {HiddenPrev}},{"Weight", {Weight}}};
    
    if (Bias != nullptr) {
      ins["Bias"] = {Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Gate"][0],outs["ResetHiddenPrev"][0],outs["Hidden"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fake_channel_wise_quantize_dequantize_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_channel_wise_quantize_dequantize_abs_max";
    platform::RecordEvent op_type_record_event("fake_channel_wise_quantize_dequantize_abs_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    auto OutScale = GetVarBaseFromArgs(op_type, "OutScale", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}},{"OutScale", {OutScale}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["OutScale"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sampling_id(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sampling_id";
    platform::RecordEvent op_type_record_event("sampling_id pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unsqueeze2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unsqueeze2";
    platform::RecordEvent op_type_record_event("unsqueeze2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (ins.count("X") && outs.count("Out")) {
      HandleViewBetweenInputAndOutput(ins["X"][0], outs["Out"][0]);
    }
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unsqueeze2_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unsqueeze2";
    platform::RecordEvent op_type_record_event("unsqueeze2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XShape"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_transfer_dtype(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "transfer_dtype";
    platform::RecordEvent op_type_record_event("transfer_dtype pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_average_accumulates(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "average_accumulates";
    platform::RecordEvent op_type_record_event("average_accumulates pybind_imperative_func");
    
    auto param = GetVarBaseFromArgs(op_type, "param", args, 0, false);
    auto in_sum_1 = GetVarBaseFromArgs(op_type, "in_sum_1", args, 1, false);
    auto in_sum_2 = GetVarBaseFromArgs(op_type, "in_sum_2", args, 2, false);
    auto in_sum_3 = GetVarBaseFromArgs(op_type, "in_sum_3", args, 3, false);
    auto in_num_accumulates = GetVarBaseFromArgs(op_type, "in_num_accumulates", args, 4, false);
    auto in_old_num_accumulates = GetVarBaseFromArgs(op_type, "in_old_num_accumulates", args, 5, false);
    auto in_num_updates = GetVarBaseFromArgs(op_type, "in_num_updates", args, 6, false);
    auto out_sum_1 = GetVarBaseFromArgs(op_type, "out_sum_1", args, 7, false);
    auto out_sum_2 = GetVarBaseFromArgs(op_type, "out_sum_2", args, 8, false);
    auto out_sum_3 = GetVarBaseFromArgs(op_type, "out_sum_3", args, 9, false);
    auto out_num_accumulates = GetVarBaseFromArgs(op_type, "out_num_accumulates", args, 10, false);
    auto out_old_num_accumulates = GetVarBaseFromArgs(op_type, "out_old_num_accumulates", args, 11, false);
    auto out_num_updates = GetVarBaseFromArgs(op_type, "out_num_updates", args, 12, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 13, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"out_sum_1", {out_sum_1}},{"out_sum_2", {out_sum_2}},{"out_sum_3", {out_sum_3}},{"out_num_accumulates", {out_num_accumulates}},{"out_old_num_accumulates", {out_old_num_accumulates}},{"out_num_updates", {out_num_updates}}};
    imperative::NameVarBaseMap ins = {{"param", {param}},{"in_sum_1", {in_sum_1}},{"in_sum_2", {in_sum_2}},{"in_sum_3", {in_sum_3}},{"in_num_accumulates", {in_num_accumulates}},{"in_old_num_accumulates", {in_old_num_accumulates}},{"in_num_updates", {in_num_updates}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["out_sum_1"][0],outs["out_sum_2"][0],outs["out_sum_3"][0],outs["out_num_accumulates"][0],outs["out_old_num_accumulates"][0],outs["out_num_updates"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_enumerate(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_enumerate";
    platform::RecordEvent op_type_record_event("sequence_enumerate pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_seqconv_eltadd_relu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_seqconv_eltadd_relu";
    platform::RecordEvent op_type_record_event("fusion_seqconv_eltadd_relu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ColMat", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Filter", {Filter}},{"Bias", {Bias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["ColMat"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bce_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bce_loss";
    platform::RecordEvent op_type_record_event("bce_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bce_loss_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bce_loss";
    platform::RecordEvent op_type_record_event("bce_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_generate_proposal_labels(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "generate_proposal_labels";
    platform::RecordEvent op_type_record_event("generate_proposal_labels pybind_imperative_func");
    
    auto RpnRois = GetVarBaseFromArgs(op_type, "RpnRois", args, 0, false);
    auto GtClasses = GetVarBaseFromArgs(op_type, "GtClasses", args, 1, false);
    auto IsCrowd = GetVarBaseFromArgs(op_type, "IsCrowd", args, 2, false);
    auto GtBoxes = GetVarBaseFromArgs(op_type, "GtBoxes", args, 3, false);
    auto ImInfo = GetVarBaseFromArgs(op_type, "ImInfo", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Rois", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LabelsInt32", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BboxTargets", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BboxInsideWeights", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BboxOutsideWeights", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MaxOverlapWithGT", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"RpnRois", {RpnRois}},{"GtClasses", {GtClasses}},{"IsCrowd", {IsCrowd}},{"GtBoxes", {GtBoxes}},{"ImInfo", {ImInfo}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Rois"][0],outs["LabelsInt32"][0],outs["BboxTargets"][0],outs["BboxInsideWeights"][0],outs["BboxOutsideWeights"][0],outs["MaxOverlapWithGT"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_im2sequence(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "im2sequence";
    platform::RecordEvent op_type_record_event("im2sequence pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_isinf(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "isinf";
    platform::RecordEvent op_type_record_event("isinf pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_logcumsumexp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logcumsumexp";
    platform::RecordEvent op_type_record_event("logcumsumexp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_adagrad(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "adagrad";
    platform::RecordEvent op_type_record_event("adagrad pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto Moment = GetVarBaseFromArgs(op_type, "Moment", args, 2, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 3, false);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 4, false);
    auto MomentOut = GetVarBaseFromArgs(op_type, "MomentOut", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"MomentOut", {MomentOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"Moment", {Moment}},{"LearningRate", {LearningRate}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["MomentOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_linear_chain_crf(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "linear_chain_crf";
    platform::RecordEvent op_type_record_event("linear_chain_crf pybind_imperative_func");
    
    auto Emission = GetVarBaseFromArgs(op_type, "Emission", args, 0, false);
    auto Transition = GetVarBaseFromArgs(op_type, "Transition", args, 1, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 2, false);
    auto Length = GetVarBaseFromArgs(op_type, "Length", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Alpha", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"EmissionExps", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TransitionExps", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LogLikelihood", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Emission", {Emission}},{"Transition", {Transition}},{"Label", {Label}}};
    
    if (Length != nullptr) {
      ins["Length"] = {Length};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Alpha"][0],outs["EmissionExps"][0],outs["TransitionExps"][0],outs["LogLikelihood"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_retinanet_target_assign(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "retinanet_target_assign";
    platform::RecordEvent op_type_record_event("retinanet_target_assign pybind_imperative_func");
    
    auto Anchor = GetVarBaseFromArgs(op_type, "Anchor", args, 0, false);
    auto GtBoxes = GetVarBaseFromArgs(op_type, "GtBoxes", args, 1, false);
    auto GtLabels = GetVarBaseFromArgs(op_type, "GtLabels", args, 2, false);
    auto IsCrowd = GetVarBaseFromArgs(op_type, "IsCrowd", args, 3, false);
    auto ImInfo = GetVarBaseFromArgs(op_type, "ImInfo", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"LocationIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ScoreIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TargetBBox", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TargetLabel", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BBoxInsideWeight", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ForegroundNumber", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Anchor", {Anchor}},{"GtBoxes", {GtBoxes}},{"GtLabels", {GtLabels}},{"IsCrowd", {IsCrowd}},{"ImInfo", {ImInfo}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["LocationIndex"][0],outs["ScoreIndex"][0],outs["TargetBBox"][0],outs["TargetLabel"][0],outs["BBoxInsideWeight"][0],outs["ForegroundNumber"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_group(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_group";
    platform::RecordEvent op_type_record_event("fusion_group pybind_imperative_func");
    
    auto Inputs = GetVarBaseListFromArgs(op_type, "Inputs", args, 0, false);
    auto OutsNum = GetUnsignedLongFromArgs(op_type, "OutsNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Outs", ConstructDuplicableOutput(OutsNum)}};
    imperative::NameVarBaseMap ins = {{"Inputs", Inputs}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Outs"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_teacher_student_sigmoid_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "teacher_student_sigmoid_loss";
    platform::RecordEvent op_type_record_event("teacher_student_sigmoid_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Y"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_random_crop(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "random_crop";
    platform::RecordEvent op_type_record_event("random_crop pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Seed = GetVarBaseFromArgs(op_type, "Seed", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SeedOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Seed", {Seed}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["SeedOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lookup_table_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lookup_table_v2";
    platform::RecordEvent op_type_record_event("lookup_table_v2 pybind_imperative_func");
    
    auto W = GetVarBaseFromArgs(op_type, "W", args, 0, false);
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"W", {W}},{"Ids", {Ids}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_fmax(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_fmax";
    platform::RecordEvent op_type_record_event("elementwise_fmax pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_index_add(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "index_add";
    platform::RecordEvent op_type_record_event("index_add pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    auto AddValue = GetVarBaseFromArgs(op_type, "AddValue", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Index", {Index}},{"AddValue", {AddValue}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_index_add_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "index_add";
    platform::RecordEvent op_type_record_event("index_add pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    auto AddValue = GetVarBaseFromArgs(op_type, "AddValue", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Index", {Index}},{"AddValue", {AddValue}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_graph_sample_neighbors(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "graph_sample_neighbors";
    platform::RecordEvent op_type_record_event("graph_sample_neighbors pybind_imperative_func");
    
    auto Row = GetVarBaseFromArgs(op_type, "Row", args, 0, false);
    auto Col_Ptr = GetVarBaseFromArgs(op_type, "Col_Ptr", args, 1, false);
    auto X = GetVarBaseFromArgs(op_type, "X", args, 2, false);
    auto Eids = GetVarBaseFromArgs(op_type, "Eids", args, 3, true);
    auto Perm_Buffer = GetVarBaseFromArgs(op_type, "Perm_Buffer", args, 4, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out_Count", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out_Eids", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Row", {Row}},{"Col_Ptr", {Col_Ptr}},{"X", {X}}};
    
    if (Eids != nullptr) {
      ins["Eids"] = {Eids};
    }

    if (Perm_Buffer != nullptr) {
      ins["Perm_Buffer"] = {Perm_Buffer};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Out_Count"][0],outs["Out_Eids"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_detection_map(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "detection_map";
    platform::RecordEvent op_type_record_event("detection_map pybind_imperative_func");
    
    auto DetectRes = GetVarBaseFromArgs(op_type, "DetectRes", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"AccumPosCount", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AccumTruePos", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AccumFalsePos", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MAP", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"DetectRes", {DetectRes}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["AccumPosCount"][0],outs["AccumTruePos"][0],outs["AccumFalsePos"][0],outs["MAP"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_l1_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "l1_norm";
    platform::RecordEvent op_type_record_event("l1_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sqrt(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sqrt";
    platform::RecordEvent op_type_record_event("sqrt pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sqrt_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sqrt";
    platform::RecordEvent op_type_record_event("sqrt pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_elemwise_activation(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_elemwise_activation";
    platform::RecordEvent op_type_record_event("fused_elemwise_activation pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"IntermediateOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["IntermediateOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_slogdeterminant(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "slogdeterminant";
    platform::RecordEvent op_type_record_event("slogdeterminant pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_share_buffer(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "share_buffer";
    platform::RecordEvent op_type_record_event("share_buffer pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    auto XOutNum = GetUnsignedLongFromArgs(op_type, "XOutNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)},{"XOut", ConstructDuplicableOutput(XOutNum)}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"],outs["XOut"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_poisson(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "poisson";
    platform::RecordEvent op_type_record_event("poisson pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bitwise_and(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bitwise_and";
    platform::RecordEvent op_type_record_event("bitwise_and pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_diag_embed(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "diag_embed";
    platform::RecordEvent op_type_record_event("diag_embed pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_check_memory_continue(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "check_memory_continue";
    platform::RecordEvent op_type_record_event("check_memory_continue pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)},{"XOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"],outs["XOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unbind(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unbind";
    platform::RecordEvent op_type_record_event("unbind pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_dropout(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dropout";
    platform::RecordEvent op_type_record_event("dropout pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Mask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Mask"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_beam_search(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "beam_search";
    platform::RecordEvent op_type_record_event("beam_search pybind_imperative_func");
    
    auto pre_ids = GetVarBaseFromArgs(op_type, "pre_ids", args, 0, false);
    auto pre_scores = GetVarBaseFromArgs(op_type, "pre_scores", args, 1, false);
    auto scores = GetVarBaseFromArgs(op_type, "scores", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"selected_ids", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"selected_scores", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"pre_ids", {pre_ids}},{"pre_scores", {pre_scores}},{"scores", {scores}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["selected_ids"][0],outs["selected_scores"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_moving_average_abs_max_scale(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "moving_average_abs_max_scale";
    platform::RecordEvent op_type_record_event("moving_average_abs_max_scale pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto InAccum = GetVarBaseFromArgs(op_type, "InAccum", args, 1, true);
    auto InState = GetVarBaseFromArgs(op_type, "InState", args, 2, true);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 3, true);
    auto OutScale = GetVarBaseFromArgs(op_type, "OutScale", args, 4, false);
    auto OutState = GetVarBaseFromArgs(op_type, "OutState", args, 5, true);
    auto OutAccum = GetVarBaseFromArgs(op_type, "OutAccum", args, 6, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 7, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"OutScale", {OutScale}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (InAccum != nullptr) {
      ins["InAccum"] = {InAccum};
    }

    if (InState != nullptr) {
      ins["InState"] = {InState};
    }

    outs["Out"] = {Out};

    outs["OutState"] = {OutState};

    outs["OutAccum"] = {OutAccum};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["OutScale"][0],outs["OutState"][0],outs["OutAccum"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_greater_than(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "greater_than";
    platform::RecordEvent op_type_record_event("greater_than pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_log_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "log_loss";
    platform::RecordEvent op_type_record_event("log_loss pybind_imperative_func");
    
    auto Predicted = GetVarBaseFromArgs(op_type, "Predicted", args, 0, false);
    auto Labels = GetVarBaseFromArgs(op_type, "Labels", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Predicted", {Predicted}},{"Labels", {Labels}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Loss"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_kron(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "kron";
    platform::RecordEvent op_type_record_event("kron pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sigmoid_focal_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sigmoid_focal_loss";
    platform::RecordEvent op_type_record_event("sigmoid_focal_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    auto FgNum = GetVarBaseFromArgs(op_type, "FgNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}},{"FgNum", {FgNum}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_rmsprop(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "rmsprop";
    platform::RecordEvent op_type_record_event("rmsprop pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto MeanSquare = GetVarBaseFromArgs(op_type, "MeanSquare", args, 1, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 2, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 3, false);
    auto Moment = GetVarBaseFromArgs(op_type, "Moment", args, 4, false);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 5, false);
    auto MomentOut = GetVarBaseFromArgs(op_type, "MomentOut", args, 6, false);
    auto MeanSquareOut = GetVarBaseFromArgs(op_type, "MeanSquareOut", args, 7, false);
    auto MeanGradOut = GetVarBaseFromArgs(op_type, "MeanGradOut", args, 8, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 9, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"MomentOut", {MomentOut}},{"MeanSquareOut", {MeanSquareOut}},{"MeanGradOut", {MeanGradOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"MeanSquare", {MeanSquare}},{"LearningRate", {LearningRate}},{"Grad", {Grad}},{"Moment", {Moment}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["MomentOut"][0],outs["MeanSquareOut"][0],outs["MeanGradOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_conv2d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conv2d";
    platform::RecordEvent op_type_record_event("conv2d pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Filter", {Filter}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Output"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_graph_reindex(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "graph_reindex";
    platform::RecordEvent op_type_record_event("graph_reindex pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Neighbors = GetVarBaseFromArgs(op_type, "Neighbors", args, 1, false);
    auto Count = GetVarBaseFromArgs(op_type, "Count", args, 2, false);
    auto HashTable_Value = GetVarBaseFromArgs(op_type, "HashTable_Value", args, 3, true);
    auto HashTable_Index = GetVarBaseFromArgs(op_type, "HashTable_Index", args, 4, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Reindex_Src", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Reindex_Dst", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out_Nodes", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Neighbors", {Neighbors}},{"Count", {Count}}};
    
    if (HashTable_Value != nullptr) {
      ins["HashTable_Value"] = {HashTable_Value};
    }

    if (HashTable_Index != nullptr) {
      ins["HashTable_Index"] = {HashTable_Index};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Reindex_Src"][0],outs["Reindex_Dst"][0],outs["Out_Nodes"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_uniform_random_inplace(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "uniform_random_inplace";
    platform::RecordEvent op_type_record_event("uniform_random_inplace pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_uniform_random_inplace_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "uniform_random_inplace";
    platform::RecordEvent op_type_record_event("uniform_random_inplace pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_maxout(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "maxout";
    platform::RecordEvent op_type_record_event("maxout pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lstsq(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lstsq";
    platform::RecordEvent op_type_record_event("lstsq pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Solution", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Residuals", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Rank", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SingularValues", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Solution"][0],outs["Residuals"][0],outs["Rank"][0],outs["SingularValues"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_linear_interp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "linear_interp";
    platform::RecordEvent op_type_record_event("linear_interp pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto OutSize = GetVarBaseFromArgs(op_type, "OutSize", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (OutSize != nullptr) {
      ins["OutSize"] = {OutSize};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_graph_khop_sampler(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "graph_khop_sampler";
    platform::RecordEvent op_type_record_event("graph_khop_sampler pybind_imperative_func");
    
    auto Row = GetVarBaseFromArgs(op_type, "Row", args, 0, false);
    auto Eids = GetVarBaseFromArgs(op_type, "Eids", args, 1, true);
    auto Col_Ptr = GetVarBaseFromArgs(op_type, "Col_Ptr", args, 2, false);
    auto X = GetVarBaseFromArgs(op_type, "X", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out_Src", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out_Dst", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Sample_Index", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Reindex_X", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out_Eids", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Row", {Row}},{"Col_Ptr", {Col_Ptr}},{"X", {X}}};
    
    if (Eids != nullptr) {
      ins["Eids"] = {Eids};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out_Src"][0],outs["Out_Dst"][0],outs["Sample_Index"][0],outs["Reindex_X"][0],outs["Out_Eids"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_put_along_axis(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "put_along_axis";
    platform::RecordEvent op_type_record_event("put_along_axis pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    auto Value = GetVarBaseFromArgs(op_type, "Value", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Result", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Index", {Index}},{"Value", {Value}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Result"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_put_along_axis_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "put_along_axis";
    platform::RecordEvent op_type_record_event("put_along_axis pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    auto Value = GetVarBaseFromArgs(op_type, "Value", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      Input->IsLeaf() && !Input->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", Input->Name()));
    Input->BumpInplaceVersion();
    VLOG(3) << "Var(" << Input->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Result", {Input}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Index", {Index}},{"Value", {Value}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"Input", "Result"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Result"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_auc(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "auc";
    platform::RecordEvent op_type_record_event("auc pybind_imperative_func");
    
    auto Predict = GetVarBaseFromArgs(op_type, "Predict", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    auto StatPos = GetVarBaseFromArgs(op_type, "StatPos", args, 2, false);
    auto StatNeg = GetVarBaseFromArgs(op_type, "StatNeg", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"AUC", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"StatPosOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"StatNegOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Predict", {Predict}},{"Label", {Label}},{"StatPos", {StatPos}},{"StatNeg", {StatNeg}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["AUC"][0],outs["StatPosOut"][0],outs["StatNegOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_logical_or(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logical_or";
    platform::RecordEvent op_type_record_event("logical_or pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_batch_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "batch_norm";
    platform::RecordEvent op_type_record_event("batch_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    auto Mean = GetVarBaseFromArgs(op_type, "Mean", args, 3, false);
    auto Variance = GetVarBaseFromArgs(op_type, "Variance", args, 4, false);
    auto MomentumTensor = GetVarBaseFromArgs(op_type, "MomentumTensor", args, 5, true);
    auto MeanOut = GetVarBaseFromArgs(op_type, "MeanOut", args, 6, false);
    auto VarianceOut = GetVarBaseFromArgs(op_type, "VarianceOut", args, 7, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 8, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MeanOut", {MeanOut}},{"VarianceOut", {VarianceOut}},{"SavedMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReserveSpace", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}},{"Bias", {Bias}},{"Mean", {Mean}},{"Variance", {Variance}}};
    
    if (MomentumTensor != nullptr) {
      ins["MomentumTensor"] = {MomentumTensor};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["MeanOut"][0],outs["VarianceOut"][0],outs["SavedMean"][0],outs["SavedVariance"][0],outs["ReserveSpace"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_add(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_add";
    platform::RecordEvent op_type_record_event("elementwise_add pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_add_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_add";
    platform::RecordEvent op_type_record_event("elementwise_add pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Out"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["Out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyMethodDef ExtestMethods[] = {
  {"reshape2", (PyCFunction)(void(*)(void))imperative_reshape2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape2 in dygraph."},
  {"reshape2_", (PyCFunction)(void(*)(void))imperative_reshape2_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape2_ in dygraph."},
  {"reduce_any", (PyCFunction)(void(*)(void))imperative_reduce_any, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reduce_any in dygraph."},
  {"limit_by_capacity", (PyCFunction)(void(*)(void))imperative_limit_by_capacity, METH_VARARGS | METH_KEYWORDS, "C++ interface function for limit_by_capacity in dygraph."},
  {"unstack", (PyCFunction)(void(*)(void))imperative_unstack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unstack in dygraph."},
  {"scatter_nd_add", (PyCFunction)(void(*)(void))imperative_scatter_nd_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scatter_nd_add in dygraph."},
  {"sequence_reshape", (PyCFunction)(void(*)(void))imperative_sequence_reshape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_reshape in dygraph."},
  {"bilateral_slice", (PyCFunction)(void(*)(void))imperative_bilateral_slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilateral_slice in dygraph."},
  {"fill_any_like", (PyCFunction)(void(*)(void))imperative_fill_any_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_any_like in dygraph."},
  {"empty", (PyCFunction)(void(*)(void))imperative_empty, METH_VARARGS | METH_KEYWORDS, "C++ interface function for empty in dygraph."},
  {"pad_constant_like", (PyCFunction)(void(*)(void))imperative_pad_constant_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad_constant_like in dygraph."},
  {"pool2d", (PyCFunction)(void(*)(void))imperative_pool2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pool2d in dygraph."},
  {"size", (PyCFunction)(void(*)(void))imperative_size, METH_VARARGS | METH_KEYWORDS, "C++ interface function for size in dygraph."},
  {"imag", (PyCFunction)(void(*)(void))imperative_imag, METH_VARARGS | METH_KEYWORDS, "C++ interface function for imag in dygraph."},
  {"pull_gpups_sparse", (PyCFunction)(void(*)(void))imperative_pull_gpups_sparse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pull_gpups_sparse in dygraph."},
  {"eigh", (PyCFunction)(void(*)(void))imperative_eigh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigh in dygraph."},
  {"stack", (PyCFunction)(void(*)(void))imperative_stack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stack in dygraph."},
  {"dgc_momentum", (PyCFunction)(void(*)(void))imperative_dgc_momentum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dgc_momentum in dygraph."},
  {"lamb", (PyCFunction)(void(*)(void))imperative_lamb, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lamb in dygraph."},
  {"generate_proposals_v2", (PyCFunction)(void(*)(void))imperative_generate_proposals_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for generate_proposals_v2 in dygraph."},
  {"bitwise_or", (PyCFunction)(void(*)(void))imperative_bitwise_or, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_or in dygraph."},
  {"gru_unit", (PyCFunction)(void(*)(void))imperative_gru_unit, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gru_unit in dygraph."},
  {"fake_channel_wise_quantize_dequantize_abs_max", (PyCFunction)(void(*)(void))imperative_fake_channel_wise_quantize_dequantize_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_channel_wise_quantize_dequantize_abs_max in dygraph."},
  {"sampling_id", (PyCFunction)(void(*)(void))imperative_sampling_id, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sampling_id in dygraph."},
  {"unsqueeze2", (PyCFunction)(void(*)(void))imperative_unsqueeze2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unsqueeze2 in dygraph."},
  {"unsqueeze2_", (PyCFunction)(void(*)(void))imperative_unsqueeze2_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unsqueeze2_ in dygraph."},
  {"transfer_dtype", (PyCFunction)(void(*)(void))imperative_transfer_dtype, METH_VARARGS | METH_KEYWORDS, "C++ interface function for transfer_dtype in dygraph."},
  {"average_accumulates", (PyCFunction)(void(*)(void))imperative_average_accumulates, METH_VARARGS | METH_KEYWORDS, "C++ interface function for average_accumulates in dygraph."},
  {"sequence_enumerate", (PyCFunction)(void(*)(void))imperative_sequence_enumerate, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_enumerate in dygraph."},
  {"fusion_seqconv_eltadd_relu", (PyCFunction)(void(*)(void))imperative_fusion_seqconv_eltadd_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_seqconv_eltadd_relu in dygraph."},
  {"bce_loss", (PyCFunction)(void(*)(void))imperative_bce_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bce_loss in dygraph."},
  {"bce_loss_", (PyCFunction)(void(*)(void))imperative_bce_loss_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bce_loss_ in dygraph."},
  {"generate_proposal_labels", (PyCFunction)(void(*)(void))imperative_generate_proposal_labels, METH_VARARGS | METH_KEYWORDS, "C++ interface function for generate_proposal_labels in dygraph."},
  {"im2sequence", (PyCFunction)(void(*)(void))imperative_im2sequence, METH_VARARGS | METH_KEYWORDS, "C++ interface function for im2sequence in dygraph."},
  {"isinf", (PyCFunction)(void(*)(void))imperative_isinf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isinf in dygraph."},
  {"logcumsumexp", (PyCFunction)(void(*)(void))imperative_logcumsumexp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logcumsumexp in dygraph."},
  {"adagrad", (PyCFunction)(void(*)(void))imperative_adagrad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adagrad in dygraph."},
  {"linear_chain_crf", (PyCFunction)(void(*)(void))imperative_linear_chain_crf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linear_chain_crf in dygraph."},
  {"retinanet_target_assign", (PyCFunction)(void(*)(void))imperative_retinanet_target_assign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for retinanet_target_assign in dygraph."},
  {"fusion_group", (PyCFunction)(void(*)(void))imperative_fusion_group, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_group in dygraph."},
  {"teacher_student_sigmoid_loss", (PyCFunction)(void(*)(void))imperative_teacher_student_sigmoid_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for teacher_student_sigmoid_loss in dygraph."},
  {"random_crop", (PyCFunction)(void(*)(void))imperative_random_crop, METH_VARARGS | METH_KEYWORDS, "C++ interface function for random_crop in dygraph."},
  {"lookup_table_v2", (PyCFunction)(void(*)(void))imperative_lookup_table_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lookup_table_v2 in dygraph."},
  {"elementwise_fmax", (PyCFunction)(void(*)(void))imperative_elementwise_fmax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_fmax in dygraph."},
  {"index_add", (PyCFunction)(void(*)(void))imperative_index_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_add in dygraph."},
  {"index_add_", (PyCFunction)(void(*)(void))imperative_index_add_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_add_ in dygraph."},
  {"graph_sample_neighbors", (PyCFunction)(void(*)(void))imperative_graph_sample_neighbors, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_sample_neighbors in dygraph."},
  {"detection_map", (PyCFunction)(void(*)(void))imperative_detection_map, METH_VARARGS | METH_KEYWORDS, "C++ interface function for detection_map in dygraph."},
  {"l1_norm", (PyCFunction)(void(*)(void))imperative_l1_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for l1_norm in dygraph."},
  {"sqrt", (PyCFunction)(void(*)(void))imperative_sqrt, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sqrt in dygraph."},
  {"sqrt_", (PyCFunction)(void(*)(void))imperative_sqrt_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sqrt_ in dygraph."},
  {"fused_elemwise_activation", (PyCFunction)(void(*)(void))imperative_fused_elemwise_activation, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_elemwise_activation in dygraph."},
  {"slogdeterminant", (PyCFunction)(void(*)(void))imperative_slogdeterminant, METH_VARARGS | METH_KEYWORDS, "C++ interface function for slogdeterminant in dygraph."},
  {"share_buffer", (PyCFunction)(void(*)(void))imperative_share_buffer, METH_VARARGS | METH_KEYWORDS, "C++ interface function for share_buffer in dygraph."},
  {"poisson", (PyCFunction)(void(*)(void))imperative_poisson, METH_VARARGS | METH_KEYWORDS, "C++ interface function for poisson in dygraph."},
  {"bitwise_and", (PyCFunction)(void(*)(void))imperative_bitwise_and, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_and in dygraph."},
  {"diag_embed", (PyCFunction)(void(*)(void))imperative_diag_embed, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diag_embed in dygraph."},
  {"check_memory_continue", (PyCFunction)(void(*)(void))imperative_check_memory_continue, METH_VARARGS | METH_KEYWORDS, "C++ interface function for check_memory_continue in dygraph."},
  {"unbind", (PyCFunction)(void(*)(void))imperative_unbind, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unbind in dygraph."},
  {"dropout", (PyCFunction)(void(*)(void))imperative_dropout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dropout in dygraph."},
  {"beam_search", (PyCFunction)(void(*)(void))imperative_beam_search, METH_VARARGS | METH_KEYWORDS, "C++ interface function for beam_search in dygraph."},
  {"moving_average_abs_max_scale", (PyCFunction)(void(*)(void))imperative_moving_average_abs_max_scale, METH_VARARGS | METH_KEYWORDS, "C++ interface function for moving_average_abs_max_scale in dygraph."},
  {"greater_than", (PyCFunction)(void(*)(void))imperative_greater_than, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_than in dygraph."},
  {"log_loss", (PyCFunction)(void(*)(void))imperative_log_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log_loss in dygraph."},
  {"kron", (PyCFunction)(void(*)(void))imperative_kron, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kron in dygraph."},
  {"sigmoid_focal_loss", (PyCFunction)(void(*)(void))imperative_sigmoid_focal_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_focal_loss in dygraph."},
  {"rmsprop", (PyCFunction)(void(*)(void))imperative_rmsprop, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rmsprop in dygraph."},
  {"conv2d", (PyCFunction)(void(*)(void))imperative_conv2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d in dygraph."},
  {"graph_reindex", (PyCFunction)(void(*)(void))imperative_graph_reindex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_reindex in dygraph."},
  {"uniform_random_inplace", (PyCFunction)(void(*)(void))imperative_uniform_random_inplace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_random_inplace in dygraph."},
  {"uniform_random_inplace_", (PyCFunction)(void(*)(void))imperative_uniform_random_inplace_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_random_inplace_ in dygraph."},
  {"maxout", (PyCFunction)(void(*)(void))imperative_maxout, METH_VARARGS | METH_KEYWORDS, "C++ interface function for maxout in dygraph."},
  {"lstsq", (PyCFunction)(void(*)(void))imperative_lstsq, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lstsq in dygraph."},
  {"linear_interp", (PyCFunction)(void(*)(void))imperative_linear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linear_interp in dygraph."},
  {"graph_khop_sampler", (PyCFunction)(void(*)(void))imperative_graph_khop_sampler, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_khop_sampler in dygraph."},
  {"put_along_axis", (PyCFunction)(void(*)(void))imperative_put_along_axis, METH_VARARGS | METH_KEYWORDS, "C++ interface function for put_along_axis in dygraph."},
  {"put_along_axis_", (PyCFunction)(void(*)(void))imperative_put_along_axis_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for put_along_axis_ in dygraph."},
  {"auc", (PyCFunction)(void(*)(void))imperative_auc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for auc in dygraph."},
  {"logical_or", (PyCFunction)(void(*)(void))imperative_logical_or, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_or in dygraph."},
  {"batch_norm", (PyCFunction)(void(*)(void))imperative_batch_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for batch_norm in dygraph."},
  {"elementwise_add", (PyCFunction)(void(*)(void))imperative_elementwise_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_add in dygraph."},
  {"elementwise_add_", (PyCFunction)(void(*)(void))imperative_elementwise_add_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_add_ in dygraph."},
  {nullptr,nullptr,0,nullptr}};

void BindOpFunctions3(pybind11::module *module) {
  auto m = module->def_submodule("ops");
  if (PyModule_AddFunctions(m.ptr(), ExtestMethods) < 0) {
    PADDLE_THROW(platform::errors::Fatal ("Add functions to core.ops failed!"));
  }

  InitOpsAttrTypeMap();}

} // namespace pybind
} // namespace paddle
