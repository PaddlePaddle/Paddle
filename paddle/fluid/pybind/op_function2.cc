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

static PyObject * imperative_smooth_l1_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "smooth_l1_loss";
    platform::RecordEvent op_type_record_event("smooth_l1_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    auto InsideWeight = GetVarBaseFromArgs(op_type, "InsideWeight", args, 2, true);
    auto OutsideWeight = GetVarBaseFromArgs(op_type, "OutsideWeight", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Diff", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    if (InsideWeight != nullptr) {
      ins["InsideWeight"] = {InsideWeight};
    }

    if (OutsideWeight != nullptr) {
      ins["OutsideWeight"] = {OutsideWeight};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Diff"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_linear_interp_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "linear_interp_v2";
    platform::RecordEvent op_type_record_event("linear_interp_v2 pybind_imperative_func");
    
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

static PyObject * imperative_momentum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "momentum";
    platform::RecordEvent op_type_record_event("momentum pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto Velocity = GetVarBaseFromArgs(op_type, "Velocity", args, 2, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 3, false);
    auto MasterParam = GetVarBaseFromArgs(op_type, "MasterParam", args, 4, true);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 5, false);
    auto VelocityOut = GetVarBaseFromArgs(op_type, "VelocityOut", args, 6, false);
    auto MasterParamOut = GetVarBaseFromArgs(op_type, "MasterParamOut", args, 7, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 8, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"VelocityOut", {VelocityOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"Velocity", {Velocity}},{"LearningRate", {LearningRate}}};
    
    if (MasterParam != nullptr) {
      ins["MasterParam"] = {MasterParam};
    }

    outs["MasterParamOut"] = {MasterParamOut};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["VelocityOut"][0],outs["MasterParamOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_temporal_shift(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "temporal_shift";
    platform::RecordEvent op_type_record_event("temporal_shift pybind_imperative_func");
    
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

static PyObject * imperative_nce(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "nce";
    platform::RecordEvent op_type_record_event("nce pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 2, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 3, true);
    auto SampleWeight = GetVarBaseFromArgs(op_type, "SampleWeight", args, 4, true);
    auto CustomDistProbs = GetVarBaseFromArgs(op_type, "CustomDistProbs", args, 5, true);
    auto CustomDistAlias = GetVarBaseFromArgs(op_type, "CustomDistAlias", args, 6, true);
    auto CustomDistAliasProbs = GetVarBaseFromArgs(op_type, "CustomDistAliasProbs", args, 7, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 8, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Cost", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SampleLogits", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SampleLabels", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Label", {Label}},{"Weight", {Weight}}};
    
    if (Bias != nullptr) {
      ins["Bias"] = {Bias};
    }

    if (SampleWeight != nullptr) {
      ins["SampleWeight"] = {SampleWeight};
    }

    if (CustomDistProbs != nullptr) {
      ins["CustomDistProbs"] = {CustomDistProbs};
    }

    if (CustomDistAlias != nullptr) {
      ins["CustomDistAlias"] = {CustomDistAlias};
    }

    if (CustomDistAliasProbs != nullptr) {
      ins["CustomDistAliasProbs"] = {CustomDistAliasProbs};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Cost"][0],outs["SampleLogits"][0],outs["SampleLabels"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_mv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "mv";
    platform::RecordEvent op_type_record_event("mv pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Vec = GetVarBaseFromArgs(op_type, "Vec", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Vec", {Vec}}};
    
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

static PyObject * imperative_dropout_nd(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dropout_nd";
    platform::RecordEvent op_type_record_event("dropout_nd pybind_imperative_func");
    
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

static PyObject * imperative_proximal_gd(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "proximal_gd";
    platform::RecordEvent op_type_record_event("proximal_gd pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"LearningRate", {LearningRate}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["ParamOut"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_memcpy_h2d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "memcpy_h2d";
    platform::RecordEvent op_type_record_event("memcpy_h2d pybind_imperative_func");
    
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

static PyObject * imperative_add_position_encoding(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "add_position_encoding";
    platform::RecordEvent op_type_record_event("add_position_encoding pybind_imperative_func");
    
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

static PyObject * imperative_cosh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cosh";
    platform::RecordEvent op_type_record_event("cosh pybind_imperative_func");
    
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

static PyObject * imperative_hash(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "hash";
    platform::RecordEvent op_type_record_event("hash pybind_imperative_func");
    
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

static PyObject * imperative_grad_add(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "grad_add";
    platform::RecordEvent op_type_record_event("grad_add pybind_imperative_func");
    
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

static PyObject * imperative_sign(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sign";
    platform::RecordEvent op_type_record_event("sign pybind_imperative_func");
    
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

static PyObject * imperative_prelu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "prelu";
    platform::RecordEvent op_type_record_event("prelu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Alpha = GetVarBaseFromArgs(op_type, "Alpha", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Alpha", {Alpha}}};
    
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

static PyObject * imperative_linspace(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "linspace";
    platform::RecordEvent op_type_record_event("linspace pybind_imperative_func");
    
    auto Start = GetVarBaseFromArgs(op_type, "Start", args, 0, false);
    auto Stop = GetVarBaseFromArgs(op_type, "Stop", args, 1, false);
    auto Num = GetVarBaseFromArgs(op_type, "Num", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Start", {Start}},{"Stop", {Stop}},{"Num", {Num}}};
    
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

static PyObject * imperative_fill_diagonal(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_diagonal";
    platform::RecordEvent op_type_record_event("fill_diagonal pybind_imperative_func");
    
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

static PyObject * imperative_fill_diagonal_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_diagonal";
    platform::RecordEvent op_type_record_event("fill_diagonal pybind_imperative_func");
    
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

static PyObject * imperative_logsigmoid(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logsigmoid";
    platform::RecordEvent op_type_record_event("logsigmoid pybind_imperative_func");
    
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

static PyObject * imperative_load_combine(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "load_combine";
    platform::RecordEvent op_type_record_event("load_combine pybind_imperative_func");
    
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {};
    
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

static PyObject * imperative_fetch_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fetch_v2";
    platform::RecordEvent op_type_record_event("fetch_v2 pybind_imperative_func");
    
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

static PyObject * imperative_randperm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "randperm";
    platform::RecordEvent op_type_record_event("randperm pybind_imperative_func");
    
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

static PyObject * imperative_sequence_scatter(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_scatter";
    platform::RecordEvent op_type_record_event("sequence_scatter pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 1, false);
    auto Updates = GetVarBaseFromArgs(op_type, "Updates", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Ids", {Ids}},{"Updates", {Updates}}};
    
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

static PyObject * imperative_relu6(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "relu6";
    platform::RecordEvent op_type_record_event("relu6 pybind_imperative_func");
    
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

static PyObject * imperative_relu6_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "relu6";
    platform::RecordEvent op_type_record_event("relu6 pybind_imperative_func");
    
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

static PyObject * imperative_partial_sum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "partial_sum";
    platform::RecordEvent op_type_record_event("partial_sum pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
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

static PyObject * imperative_sparse_add(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sparse_add";
    platform::RecordEvent op_type_record_event("sparse_add pybind_imperative_func");
    
    auto x = GetVarBaseFromArgs(op_type, "x", args, 0, false);
    auto y = GetVarBaseFromArgs(op_type, "y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"x", {x}},{"y", {y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["out"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_conv3d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conv3d";
    platform::RecordEvent op_type_record_event("conv3d pybind_imperative_func");
    
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

static PyObject * imperative_lu_unpack(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lu_unpack";
    platform::RecordEvent op_type_record_event("lu_unpack pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Pivots = GetVarBaseFromArgs(op_type, "Pivots", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Pmat", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"L", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"U", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Pivots", {Pivots}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Pmat"][0],outs["L"][0],outs["U"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lstm_unit(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lstm_unit";
    platform::RecordEvent op_type_record_event("lstm_unit pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto C_prev = GetVarBaseFromArgs(op_type, "C_prev", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"C", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"H", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"C_prev", {C_prev}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["C"][0],outs["H"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_not_equal(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "not_equal";
    platform::RecordEvent op_type_record_event("not_equal pybind_imperative_func");
    
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

static PyObject * imperative_transpose2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "transpose2";
    platform::RecordEvent op_type_record_event("transpose2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
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

static PyObject * imperative_uniform_random_batch_size_like(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "uniform_random_batch_size_like";
    platform::RecordEvent op_type_record_event("uniform_random_batch_size_like pybind_imperative_func");
    
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

static PyObject * imperative_yolo_box_head(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "yolo_box_head";
    platform::RecordEvent op_type_record_event("yolo_box_head pybind_imperative_func");
    
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

static PyObject * imperative_unfold(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unfold";
    platform::RecordEvent op_type_record_event("unfold pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
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

static PyObject * imperative_lrn(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lrn";
    platform::RecordEvent op_type_record_event("lrn pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MidOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["MidOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_isclose(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "isclose";
    platform::RecordEvent op_type_record_event("isclose pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Other = GetVarBaseFromArgs(op_type, "Other", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Other", {Other}}};
    
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

static PyObject * imperative_softmax_with_cross_entropy(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "softmax_with_cross_entropy";
    platform::RecordEvent op_type_record_event("softmax_with_cross_entropy pybind_imperative_func");
    
    auto Logits = GetVarBaseFromArgs(op_type, "Logits", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Softmax", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Logits", {Logits}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Softmax"][0],outs["Loss"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_softmax_with_cross_entropy_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "softmax_with_cross_entropy";
    platform::RecordEvent op_type_record_event("softmax_with_cross_entropy pybind_imperative_func");
    
    auto Logits = GetVarBaseFromArgs(op_type, "Logits", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      Logits->IsLeaf() && !Logits->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", Logits->Name()));
    Logits->BumpInplaceVersion();
    VLOG(3) << "Var(" << Logits->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Softmax", {Logits}},{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Logits", {Logits}},{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"Logits", "Softmax"}});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Softmax"][0],outs["Loss"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_isfinite_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "isfinite_v2";
    platform::RecordEvent op_type_record_event("isfinite_v2 pybind_imperative_func");
    
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

static PyObject * imperative_bernoulli(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bernoulli";
    platform::RecordEvent op_type_record_event("bernoulli pybind_imperative_func");
    
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

static PyObject * imperative_max_pool3d_with_index(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "max_pool3d_with_index";
    platform::RecordEvent op_type_record_event("max_pool3d_with_index pybind_imperative_func");
    
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

static PyObject * imperative_fused_seqpool_cvm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_seqpool_cvm";
    platform::RecordEvent op_type_record_event("fused_seqpool_cvm pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto CVM = GetVarBaseFromArgs(op_type, "CVM", args, 1, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"X", X},{"CVM", {CVM}}};
    
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

static PyObject * imperative_gaussian_random(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gaussian_random";
    platform::RecordEvent op_type_record_event("gaussian_random pybind_imperative_func");
    
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

static PyObject * imperative_flatten2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "flatten2";
    platform::RecordEvent op_type_record_event("flatten2 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XShape", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
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

static PyObject * imperative_flatten2_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "flatten2";
    platform::RecordEvent op_type_record_event("flatten2 pybind_imperative_func");
    
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

static PyObject * imperative_matmul(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "matmul";
    platform::RecordEvent op_type_record_event("matmul pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
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

static PyObject * imperative_cvm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cvm";
    platform::RecordEvent op_type_record_event("cvm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto CVM = GetVarBaseFromArgs(op_type, "CVM", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"CVM", {CVM}}};
    
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

static PyObject * imperative_adamax(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "adamax";
    platform::RecordEvent op_type_record_event("adamax pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 1, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 2, false);
    auto Moment = GetVarBaseFromArgs(op_type, "Moment", args, 3, false);
    auto InfNorm = GetVarBaseFromArgs(op_type, "InfNorm", args, 4, false);
    auto Beta1Pow = GetVarBaseFromArgs(op_type, "Beta1Pow", args, 5, false);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 6, false);
    auto MomentOut = GetVarBaseFromArgs(op_type, "MomentOut", args, 7, false);
    auto InfNormOut = GetVarBaseFromArgs(op_type, "InfNormOut", args, 8, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 9, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"MomentOut", {MomentOut}},{"InfNormOut", {InfNormOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"LearningRate", {LearningRate}},{"Moment", {Moment}},{"InfNorm", {InfNorm}},{"Beta1Pow", {Beta1Pow}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["MomentOut"][0],outs["InfNormOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_requantize(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "requantize";
    platform::RecordEvent op_type_record_event("requantize pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
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

static PyObject * imperative_masked_select(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "masked_select";
    platform::RecordEvent op_type_record_event("masked_select pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Mask = GetVarBaseFromArgs(op_type, "Mask", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Mask", {Mask}}};
    
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

static PyObject * imperative_range(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "range";
    platform::RecordEvent op_type_record_event("range pybind_imperative_func");
    
    auto Start = GetVarBaseFromArgs(op_type, "Start", args, 0, false);
    auto End = GetVarBaseFromArgs(op_type, "End", args, 1, false);
    auto Step = GetVarBaseFromArgs(op_type, "Step", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Start", {Start}},{"End", {End}},{"Step", {Step}}};
    
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

static PyObject * imperative_bitwise_not(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bitwise_not";
    platform::RecordEvent op_type_record_event("bitwise_not pybind_imperative_func");
    
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

static PyObject * imperative_trace(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "trace";
    platform::RecordEvent op_type_record_event("trace pybind_imperative_func");
    
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

static PyObject * imperative_multinomial(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "multinomial";
    platform::RecordEvent op_type_record_event("multinomial pybind_imperative_func");
    
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

static PyObject * imperative_sparse_conv3d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sparse_conv3d";
    platform::RecordEvent op_type_record_event("sparse_conv3d pybind_imperative_func");
    
    auto x = GetVarBaseFromArgs(op_type, "x", args, 0, false);
    auto kernel = GetVarBaseFromArgs(op_type, "kernel", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"rulebook", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"counter", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"x", {x}},{"kernel", {kernel}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["out"][0],outs["rulebook"][0],outs["counter"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_modified_huber_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "modified_huber_loss";
    platform::RecordEvent op_type_record_event("modified_huber_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"IntermediateVal", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["IntermediateVal"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_roll(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "roll";
    platform::RecordEvent op_type_record_event("roll pybind_imperative_func");
    
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

static PyObject * imperative_squared_l2_distance(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "squared_l2_distance";
    platform::RecordEvent op_type_record_event("squared_l2_distance pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"sub_result", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["sub_result"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_conv3d_transpose(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conv3d_transpose";
    platform::RecordEvent op_type_record_event("conv3d_transpose pybind_imperative_func");
    
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

static PyObject * imperative_share_data(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "share_data";
    platform::RecordEvent op_type_record_event("share_data pybind_imperative_func");
    
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

static PyObject * imperative_fake_quantize_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_quantize_abs_max";
    platform::RecordEvent op_type_record_event("fake_quantize_abs_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutScale", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
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

static PyObject * imperative_rrelu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "rrelu";
    platform::RecordEvent op_type_record_event("rrelu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Noise", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Noise"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unique_with_counts(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unique_with_counts";
    platform::RecordEvent op_type_record_event("unique_with_counts pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Index", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Count", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Index"][0],outs["Count"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill";
    platform::RecordEvent op_type_record_event("fill pybind_imperative_func");
    
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

static PyObject * imperative_concat(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "concat";
    platform::RecordEvent op_type_record_event("concat pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
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

static PyObject * imperative_fill_zeros_like(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_zeros_like";
    platform::RecordEvent op_type_record_event("fill_zeros_like pybind_imperative_func");
    
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

static PyObject * imperative_hierarchical_sigmoid(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "hierarchical_sigmoid";
    platform::RecordEvent op_type_record_event("hierarchical_sigmoid pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto W = GetVarBaseFromArgs(op_type, "W", args, 1, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 2, false);
    auto PathTable = GetVarBaseFromArgs(op_type, "PathTable", args, 3, true);
    auto PathCode = GetVarBaseFromArgs(op_type, "PathCode", args, 4, true);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 5, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"PreOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"W_Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"W", {W}},{"Label", {Label}}};
    
    if (PathTable != nullptr) {
      ins["PathTable"] = {PathTable};
    }

    if (PathCode != nullptr) {
      ins["PathCode"] = {PathCode};
    }

    if (Bias != nullptr) {
      ins["Bias"] = {Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["PreOut"][0],outs["W_Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_isinf_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "isinf_v2";
    platform::RecordEvent op_type_record_event("isinf_v2 pybind_imperative_func");
    
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

static PyObject * imperative_squeeze(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "squeeze";
    platform::RecordEvent op_type_record_event("squeeze pybind_imperative_func");
    
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

static PyObject * imperative_multiclass_nms2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "multiclass_nms2";
    platform::RecordEvent op_type_record_event("multiclass_nms2 pybind_imperative_func");
    
    auto BBoxes = GetVarBaseFromArgs(op_type, "BBoxes", args, 0, false);
    auto Scores = GetVarBaseFromArgs(op_type, "Scores", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Index", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"BBoxes", {BBoxes}},{"Scores", {Scores}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Index"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_bpr_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bpr_loss";
    platform::RecordEvent op_type_record_event("bpr_loss pybind_imperative_func");
    
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

static PyObject * imperative_fft_c2c(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fft_c2c";
    platform::RecordEvent op_type_record_event("fft_c2c pybind_imperative_func");
    
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

static PyObject * imperative_bicubic_interp_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bicubic_interp_v2";
    platform::RecordEvent op_type_record_event("bicubic_interp_v2 pybind_imperative_func");
    
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

static PyObject * imperative_angle(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "angle";
    platform::RecordEvent op_type_record_event("angle pybind_imperative_func");
    
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

static PyObject * imperative_reshape(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reshape";
    platform::RecordEvent op_type_record_event("reshape pybind_imperative_func");
    
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

static PyObject * imperative_reshape_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reshape";
    platform::RecordEvent op_type_record_event("reshape pybind_imperative_func");
    
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

static PyObject * imperative_coalesce_tensor(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "coalesce_tensor";
    platform::RecordEvent op_type_record_event("coalesce_tensor pybind_imperative_func");
    
    auto Input = GetVarBaseListFromArgs(op_type, "Input", args, 0, false);
    auto Output = GetVarBaseListFromArgs(op_type, "Output", args, 1, false);
    auto FusedOutput = GetVarBaseFromArgs(op_type, "FusedOutput", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", Output},{"FusedOutput", {FusedOutput}}};
    imperative::NameVarBaseMap ins = {{"Input", Input}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Output"],outs["FusedOutput"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_roi_align(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "roi_align";
    platform::RecordEvent op_type_record_event("roi_align pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ROIs = GetVarBaseFromArgs(op_type, "ROIs", args, 1, false);
    auto RoisNum = GetVarBaseFromArgs(op_type, "RoisNum", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"ROIs", {ROIs}}};
    
    if (RoisNum != nullptr) {
      ins["RoisNum"] = {RoisNum};
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

static PyMethodDef ExtestMethods[] = {
  {"smooth_l1_loss", (PyCFunction)(void(*)(void))imperative_smooth_l1_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for smooth_l1_loss in dygraph."},
  {"linear_interp_v2", (PyCFunction)(void(*)(void))imperative_linear_interp_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linear_interp_v2 in dygraph."},
  {"momentum", (PyCFunction)(void(*)(void))imperative_momentum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for momentum in dygraph."},
  {"temporal_shift", (PyCFunction)(void(*)(void))imperative_temporal_shift, METH_VARARGS | METH_KEYWORDS, "C++ interface function for temporal_shift in dygraph."},
  {"nce", (PyCFunction)(void(*)(void))imperative_nce, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nce in dygraph."},
  {"mv", (PyCFunction)(void(*)(void))imperative_mv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mv in dygraph."},
  {"dropout_nd", (PyCFunction)(void(*)(void))imperative_dropout_nd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dropout_nd in dygraph."},
  {"proximal_gd", (PyCFunction)(void(*)(void))imperative_proximal_gd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for proximal_gd in dygraph."},
  {"memcpy_h2d", (PyCFunction)(void(*)(void))imperative_memcpy_h2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memcpy_h2d in dygraph."},
  {"add_position_encoding", (PyCFunction)(void(*)(void))imperative_add_position_encoding, METH_VARARGS | METH_KEYWORDS, "C++ interface function for add_position_encoding in dygraph."},
  {"cosh", (PyCFunction)(void(*)(void))imperative_cosh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cosh in dygraph."},
  {"hash", (PyCFunction)(void(*)(void))imperative_hash, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hash in dygraph."},
  {"grad_add", (PyCFunction)(void(*)(void))imperative_grad_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for grad_add in dygraph."},
  {"sign", (PyCFunction)(void(*)(void))imperative_sign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sign in dygraph."},
  {"prelu", (PyCFunction)(void(*)(void))imperative_prelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prelu in dygraph."},
  {"linspace", (PyCFunction)(void(*)(void))imperative_linspace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for linspace in dygraph."},
  {"fill_diagonal", (PyCFunction)(void(*)(void))imperative_fill_diagonal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal in dygraph."},
  {"fill_diagonal_", (PyCFunction)(void(*)(void))imperative_fill_diagonal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_ in dygraph."},
  {"logsigmoid", (PyCFunction)(void(*)(void))imperative_logsigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logsigmoid in dygraph."},
  {"load_combine", (PyCFunction)(void(*)(void))imperative_load_combine, METH_VARARGS | METH_KEYWORDS, "C++ interface function for load_combine in dygraph."},
  {"fetch_v2", (PyCFunction)(void(*)(void))imperative_fetch_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fetch_v2 in dygraph."},
  {"randperm", (PyCFunction)(void(*)(void))imperative_randperm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for randperm in dygraph."},
  {"sequence_scatter", (PyCFunction)(void(*)(void))imperative_sequence_scatter, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_scatter in dygraph."},
  {"relu6", (PyCFunction)(void(*)(void))imperative_relu6, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu6 in dygraph."},
  {"relu6_", (PyCFunction)(void(*)(void))imperative_relu6_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for relu6_ in dygraph."},
  {"partial_sum", (PyCFunction)(void(*)(void))imperative_partial_sum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for partial_sum in dygraph."},
  {"sparse_add", (PyCFunction)(void(*)(void))imperative_sparse_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_add in dygraph."},
  {"conv3d", (PyCFunction)(void(*)(void))imperative_conv3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv3d in dygraph."},
  {"lu_unpack", (PyCFunction)(void(*)(void))imperative_lu_unpack, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lu_unpack in dygraph."},
  {"lstm_unit", (PyCFunction)(void(*)(void))imperative_lstm_unit, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lstm_unit in dygraph."},
  {"not_equal", (PyCFunction)(void(*)(void))imperative_not_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for not_equal in dygraph."},
  {"transpose2", (PyCFunction)(void(*)(void))imperative_transpose2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for transpose2 in dygraph."},
  {"uniform_random_batch_size_like", (PyCFunction)(void(*)(void))imperative_uniform_random_batch_size_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for uniform_random_batch_size_like in dygraph."},
  {"yolo_box_head", (PyCFunction)(void(*)(void))imperative_yolo_box_head, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolo_box_head in dygraph."},
  {"unfold", (PyCFunction)(void(*)(void))imperative_unfold, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unfold in dygraph."},
  {"lrn", (PyCFunction)(void(*)(void))imperative_lrn, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lrn in dygraph."},
  {"isclose", (PyCFunction)(void(*)(void))imperative_isclose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isclose in dygraph."},
  {"softmax_with_cross_entropy", (PyCFunction)(void(*)(void))imperative_softmax_with_cross_entropy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax_with_cross_entropy in dygraph."},
  {"softmax_with_cross_entropy_", (PyCFunction)(void(*)(void))imperative_softmax_with_cross_entropy_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softmax_with_cross_entropy_ in dygraph."},
  {"isfinite_v2", (PyCFunction)(void(*)(void))imperative_isfinite_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isfinite_v2 in dygraph."},
  {"bernoulli", (PyCFunction)(void(*)(void))imperative_bernoulli, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bernoulli in dygraph."},
  {"max_pool3d_with_index", (PyCFunction)(void(*)(void))imperative_max_pool3d_with_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max_pool3d_with_index in dygraph."},
  {"fused_seqpool_cvm", (PyCFunction)(void(*)(void))imperative_fused_seqpool_cvm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_seqpool_cvm in dygraph."},
  {"gaussian_random", (PyCFunction)(void(*)(void))imperative_gaussian_random, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gaussian_random in dygraph."},
  {"flatten2", (PyCFunction)(void(*)(void))imperative_flatten2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten2 in dygraph."},
  {"flatten2_", (PyCFunction)(void(*)(void))imperative_flatten2_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten2_ in dygraph."},
  {"matmul", (PyCFunction)(void(*)(void))imperative_matmul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matmul in dygraph."},
  {"cvm", (PyCFunction)(void(*)(void))imperative_cvm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cvm in dygraph."},
  {"adamax", (PyCFunction)(void(*)(void))imperative_adamax, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adamax in dygraph."},
  {"requantize", (PyCFunction)(void(*)(void))imperative_requantize, METH_VARARGS | METH_KEYWORDS, "C++ interface function for requantize in dygraph."},
  {"masked_select", (PyCFunction)(void(*)(void))imperative_masked_select, METH_VARARGS | METH_KEYWORDS, "C++ interface function for masked_select in dygraph."},
  {"range", (PyCFunction)(void(*)(void))imperative_range, METH_VARARGS | METH_KEYWORDS, "C++ interface function for range in dygraph."},
  {"bitwise_not", (PyCFunction)(void(*)(void))imperative_bitwise_not, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bitwise_not in dygraph."},
  {"trace", (PyCFunction)(void(*)(void))imperative_trace, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trace in dygraph."},
  {"multinomial", (PyCFunction)(void(*)(void))imperative_multinomial, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multinomial in dygraph."},
  {"sparse_conv3d", (PyCFunction)(void(*)(void))imperative_sparse_conv3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_conv3d in dygraph."},
  {"modified_huber_loss", (PyCFunction)(void(*)(void))imperative_modified_huber_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for modified_huber_loss in dygraph."},
  {"roll", (PyCFunction)(void(*)(void))imperative_roll, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roll in dygraph."},
  {"squared_l2_distance", (PyCFunction)(void(*)(void))imperative_squared_l2_distance, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squared_l2_distance in dygraph."},
  {"conv3d_transpose", (PyCFunction)(void(*)(void))imperative_conv3d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv3d_transpose in dygraph."},
  {"share_data", (PyCFunction)(void(*)(void))imperative_share_data, METH_VARARGS | METH_KEYWORDS, "C++ interface function for share_data in dygraph."},
  {"fake_quantize_abs_max", (PyCFunction)(void(*)(void))imperative_fake_quantize_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_abs_max in dygraph."},
  {"rrelu", (PyCFunction)(void(*)(void))imperative_rrelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rrelu in dygraph."},
  {"unique_with_counts", (PyCFunction)(void(*)(void))imperative_unique_with_counts, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unique_with_counts in dygraph."},
  {"fill", (PyCFunction)(void(*)(void))imperative_fill, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill in dygraph."},
  {"concat", (PyCFunction)(void(*)(void))imperative_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for concat in dygraph."},
  {"fill_zeros_like", (PyCFunction)(void(*)(void))imperative_fill_zeros_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_zeros_like in dygraph."},
  {"hierarchical_sigmoid", (PyCFunction)(void(*)(void))imperative_hierarchical_sigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hierarchical_sigmoid in dygraph."},
  {"isinf_v2", (PyCFunction)(void(*)(void))imperative_isinf_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isinf_v2 in dygraph."},
  {"squeeze", (PyCFunction)(void(*)(void))imperative_squeeze, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze in dygraph."},
  {"multiclass_nms2", (PyCFunction)(void(*)(void))imperative_multiclass_nms2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiclass_nms2 in dygraph."},
  {"bpr_loss", (PyCFunction)(void(*)(void))imperative_bpr_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bpr_loss in dygraph."},
  {"fft_c2c", (PyCFunction)(void(*)(void))imperative_fft_c2c, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_c2c in dygraph."},
  {"bicubic_interp_v2", (PyCFunction)(void(*)(void))imperative_bicubic_interp_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bicubic_interp_v2 in dygraph."},
  {"angle", (PyCFunction)(void(*)(void))imperative_angle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for angle in dygraph."},
  {"reshape", (PyCFunction)(void(*)(void))imperative_reshape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape in dygraph."},
  {"reshape_", (PyCFunction)(void(*)(void))imperative_reshape_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reshape_ in dygraph."},
  {"coalesce_tensor", (PyCFunction)(void(*)(void))imperative_coalesce_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for coalesce_tensor in dygraph."},
  {"roi_align", (PyCFunction)(void(*)(void))imperative_roi_align, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_align in dygraph."},
  {nullptr,nullptr,0,nullptr}};

void BindOpFunctions2(pybind11::module *module) {
  auto m = module->def_submodule("ops");
  if (PyModule_AddFunctions(m.ptr(), ExtestMethods) < 0) {
    PADDLE_THROW(platform::errors::Fatal ("Add functions to core.ops failed!"));
  }

  InitOpsAttrTypeMap();}

} // namespace pybind
} // namespace paddle
