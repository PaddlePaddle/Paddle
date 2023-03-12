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

static PyObject * imperative_dirichlet(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dirichlet";
    platform::RecordEvent op_type_record_event("dirichlet pybind_imperative_func");
    
    auto Alpha = GetVarBaseFromArgs(op_type, "Alpha", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Alpha", {Alpha}}};
    
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

static PyObject * imperative_stanh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "stanh";
    platform::RecordEvent op_type_record_event("stanh pybind_imperative_func");
    
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

static PyObject * imperative_label_smooth(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "label_smooth";
    platform::RecordEvent op_type_record_event("label_smooth pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto PriorDist = GetVarBaseFromArgs(op_type, "PriorDist", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (PriorDist != nullptr) {
      ins["PriorDist"] = {PriorDist};
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

static PyObject * imperative_fold(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fold";
    platform::RecordEvent op_type_record_event("fold pybind_imperative_func");
    
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

static PyObject * imperative_merged_momentum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "merged_momentum";
    platform::RecordEvent op_type_record_event("merged_momentum pybind_imperative_func");
    
    auto Param = GetVarBaseListFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseListFromArgs(op_type, "Grad", args, 1, false);
    auto Velocity = GetVarBaseListFromArgs(op_type, "Velocity", args, 2, false);
    auto LearningRate = GetVarBaseListFromArgs(op_type, "LearningRate", args, 3, false);
    auto MasterParam = GetVarBaseListFromArgs(op_type, "MasterParam", args, 4, true);
    auto ParamOut = GetVarBaseListFromArgs(op_type, "ParamOut", args, 5, false);
    auto VelocityOut = GetVarBaseListFromArgs(op_type, "VelocityOut", args, 6, false);
    auto MasterParamOut = GetVarBaseListFromArgs(op_type, "MasterParamOut", args, 7, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 8, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", ParamOut},{"VelocityOut", VelocityOut}};
    imperative::NameVarBaseMap ins = {{"Param", Param},{"Grad", Grad},{"Velocity", Velocity},{"LearningRate", LearningRate}};
    
    if (MasterParam.size() != 0) {
      ins["MasterParam"] = MasterParam;
    }

    outs["MasterParamOut"] = MasterParamOut;

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"],outs["VelocityOut"],outs["MasterParamOut"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_ascend_trigger(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "ascend_trigger";
    platform::RecordEvent op_type_record_event("ascend_trigger pybind_imperative_func");
    
    auto FeedList = GetVarBaseListFromArgs(op_type, "FeedList", args, 0, false);
    auto FetchListNum = GetUnsignedLongFromArgs(op_type, "FetchListNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"FetchList", ConstructDuplicableOutput(FetchListNum)}};
    imperative::NameVarBaseMap ins = {{"FeedList", FeedList}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["FetchList"]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_rpn_target_assign(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "rpn_target_assign";
    platform::RecordEvent op_type_record_event("rpn_target_assign pybind_imperative_func");
    
    auto Anchor = GetVarBaseFromArgs(op_type, "Anchor", args, 0, false);
    auto GtBoxes = GetVarBaseFromArgs(op_type, "GtBoxes", args, 1, false);
    auto IsCrowd = GetVarBaseFromArgs(op_type, "IsCrowd", args, 2, false);
    auto ImInfo = GetVarBaseFromArgs(op_type, "ImInfo", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"LocationIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ScoreIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TargetBBox", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TargetLabel", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BBoxInsideWeight", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Anchor", {Anchor}},{"GtBoxes", {GtBoxes}},{"IsCrowd", {IsCrowd}},{"ImInfo", {ImInfo}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["LocationIndex"][0],outs["ScoreIndex"][0],outs["TargetBBox"][0],outs["TargetLabel"][0],outs["BBoxInsideWeight"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_feedforward(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_feedforward";
    platform::RecordEvent op_type_record_event("fused_feedforward pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Dropout1Seed = GetVarBaseFromArgs(op_type, "Dropout1Seed", args, 1, true);
    auto Dropout2Seed = GetVarBaseFromArgs(op_type, "Dropout2Seed", args, 2, true);
    auto Linear1Weight = GetVarBaseFromArgs(op_type, "Linear1Weight", args, 3, false);
    auto Linear1Bias = GetVarBaseFromArgs(op_type, "Linear1Bias", args, 4, true);
    auto Linear2Weight = GetVarBaseFromArgs(op_type, "Linear2Weight", args, 5, false);
    auto Linear2Bias = GetVarBaseFromArgs(op_type, "Linear2Bias", args, 6, true);
    auto Ln1Scale = GetVarBaseFromArgs(op_type, "Ln1Scale", args, 7, true);
    auto Ln1Bias = GetVarBaseFromArgs(op_type, "Ln1Bias", args, 8, true);
    auto Ln2Scale = GetVarBaseFromArgs(op_type, "Ln2Scale", args, 9, true);
    auto Ln2Bias = GetVarBaseFromArgs(op_type, "Ln2Bias", args, 10, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 11, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Dropout1Mask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Dropout2Mask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln1Mean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln1Variance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln2Mean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln2Variance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Linear1Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln1Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Dropout1Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Dropout2Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Linear1Weight", {Linear1Weight}},{"Linear2Weight", {Linear2Weight}}};
    
    if (Dropout1Seed != nullptr) {
      ins["Dropout1Seed"] = {Dropout1Seed};
    }

    if (Dropout2Seed != nullptr) {
      ins["Dropout2Seed"] = {Dropout2Seed};
    }

    if (Linear1Bias != nullptr) {
      ins["Linear1Bias"] = {Linear1Bias};
    }

    if (Linear2Bias != nullptr) {
      ins["Linear2Bias"] = {Linear2Bias};
    }

    if (Ln1Scale != nullptr) {
      ins["Ln1Scale"] = {Ln1Scale};
    }

    if (Ln1Bias != nullptr) {
      ins["Ln1Bias"] = {Ln1Bias};
    }

    if (Ln2Scale != nullptr) {
      ins["Ln2Scale"] = {Ln2Scale};
    }

    if (Ln2Bias != nullptr) {
      ins["Ln2Bias"] = {Ln2Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Dropout1Mask"][0],outs["Dropout2Mask"][0],outs["Ln1Mean"][0],outs["Ln1Variance"][0],outs["Ln2Mean"][0],outs["Ln2Variance"][0],outs["Linear1Out"][0],outs["Ln1Out"][0],outs["Dropout1Out"][0],outs["Dropout2Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_roi_perspective_transform(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "roi_perspective_transform";
    platform::RecordEvent op_type_record_event("roi_perspective_transform pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ROIs = GetVarBaseFromArgs(op_type, "ROIs", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Mask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TransformMatrix", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out2InIdx", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out2InWeights", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"ROIs", {ROIs}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Mask"][0],outs["TransformMatrix"][0],outs["Out2InIdx"][0],outs["Out2InWeights"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_expand(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "expand";
    platform::RecordEvent op_type_record_event("expand pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ExpandTimes = GetVarBaseFromArgs(op_type, "ExpandTimes", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (ExpandTimes != nullptr) {
      ins["ExpandTimes"] = {ExpandTimes};
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

static PyObject * imperative_prroi_pool(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "prroi_pool";
    platform::RecordEvent op_type_record_event("prroi_pool pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ROIs = GetVarBaseFromArgs(op_type, "ROIs", args, 1, false);
    auto BatchRoINums = GetVarBaseFromArgs(op_type, "BatchRoINums", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"ROIs", {ROIs}}};
    
    if (BatchRoINums != nullptr) {
      ins["BatchRoINums"] = {BatchRoINums};
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

static PyObject * imperative_pool3d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pool3d";
    platform::RecordEvent op_type_record_event("pool3d pybind_imperative_func");
    
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

static PyObject * imperative_memcpy(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "memcpy";
    platform::RecordEvent op_type_record_event("memcpy pybind_imperative_func");
    
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

static PyObject * imperative_distribute_fpn_proposals(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "distribute_fpn_proposals";
    platform::RecordEvent op_type_record_event("distribute_fpn_proposals pybind_imperative_func");
    
    auto FpnRois = GetVarBaseFromArgs(op_type, "FpnRois", args, 0, false);
    auto RoisNum = GetVarBaseFromArgs(op_type, "RoisNum", args, 1, true);
    auto MultiFpnRoisNum = GetUnsignedLongFromArgs(op_type, "MultiFpnRoisNum", args, 2, false);
    auto MultiLevelRoIsNumNum = GetUnsignedLongFromArgs(op_type, "MultiLevelRoIsNumNum", args, 3, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"MultiFpnRois", ConstructDuplicableOutput(MultiFpnRoisNum)},{"RestoreIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MultiLevelRoIsNum", ConstructDuplicableOutput(MultiLevelRoIsNumNum)}};
    imperative::NameVarBaseMap ins = {{"FpnRois", {FpnRois}}};
    
    if (RoisNum != nullptr) {
      ins["RoisNum"] = {RoisNum};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["MultiFpnRois"],outs["RestoreIndex"][0],outs["MultiLevelRoIsNum"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_frame(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "frame";
    platform::RecordEvent op_type_record_event("frame pybind_imperative_func");
    
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

static PyObject * imperative_bincount(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bincount";
    platform::RecordEvent op_type_record_event("bincount pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Weights = GetVarBaseFromArgs(op_type, "Weights", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Weights != nullptr) {
      ins["Weights"] = {Weights};
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

static PyObject * imperative_shape(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "shape";
    platform::RecordEvent op_type_record_event("shape pybind_imperative_func");
    
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

static PyObject * imperative_mode(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "mode";
    platform::RecordEvent op_type_record_event("mode pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Indices", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Indices"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_group_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "group_norm";
    platform::RecordEvent op_type_record_event("group_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, true);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, true);
    auto Mean = GetVarBaseFromArgs(op_type, "Mean", args, 3, false);
    auto Variance = GetVarBaseFromArgs(op_type, "Variance", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Mean", {Mean}},{"Variance", {Variance}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Scale != nullptr) {
      ins["Scale"] = {Scale};
    }

    if (Bias != nullptr) {
      ins["Bias"] = {Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["Mean"][0],outs["Variance"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_resnet_unit(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "resnet_unit";
    platform::RecordEvent op_type_record_event("resnet_unit pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto FilterX = GetVarBaseFromArgs(op_type, "FilterX", args, 1, false);
    auto ScaleX = GetVarBaseFromArgs(op_type, "ScaleX", args, 2, false);
    auto BiasX = GetVarBaseFromArgs(op_type, "BiasX", args, 3, false);
    auto MeanX = GetVarBaseFromArgs(op_type, "MeanX", args, 4, false);
    auto VarX = GetVarBaseFromArgs(op_type, "VarX", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BitMask", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ConvX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedMeanX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedInvstdX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RunningMeanX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RunningVarX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"FilterX", {FilterX}},{"ScaleX", {ScaleX}},{"BiasX", {BiasX}},{"MeanX", {MeanX}},{"VarX", {VarX}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["BitMask"][0],outs["ConvX"][0],outs["SavedMeanX"][0],outs["SavedInvstdX"][0],outs["RunningMeanX"][0],outs["RunningVarX"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_expand_as(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_expand_as";
    platform::RecordEvent op_type_record_event("sequence_expand_as pybind_imperative_func");
    
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

static PyObject * imperative_cos_sim(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cos_sim";
    platform::RecordEvent op_type_record_event("cos_sim pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XNorm", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"YNorm", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["XNorm"][0],outs["YNorm"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_eigvals(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "eigvals";
    platform::RecordEvent op_type_record_event("eigvals pybind_imperative_func");
    
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

static PyObject * imperative_save_combine(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "save_combine";
    platform::RecordEvent op_type_record_event("save_combine pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    RETURN_PY_NONE
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_class_center_sample(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "class_center_sample";
    platform::RecordEvent op_type_record_event("class_center_sample pybind_imperative_func");
    
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"RemappedLabel", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SampledLocalClassCenter", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Label", {Label}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["RemappedLabel"][0],outs["SampledLocalClassCenter"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_fmin(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_fmin";
    platform::RecordEvent op_type_record_event("elementwise_fmin pybind_imperative_func");
    
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

static PyObject * imperative_read_file(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "read_file";
    platform::RecordEvent op_type_record_event("read_file pybind_imperative_func");
    
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

static PyObject * imperative_isfinite(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "isfinite";
    platform::RecordEvent op_type_record_event("isfinite pybind_imperative_func");
    
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

static PyObject * imperative_arg_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "arg_max";
    platform::RecordEvent op_type_record_event("arg_max pybind_imperative_func");
    
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

static PyObject * imperative_equal(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "equal";
    platform::RecordEvent op_type_record_event("equal pybind_imperative_func");
    
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

static PyObject * imperative_fake_dequantize_max_abs(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_dequantize_max_abs";
    platform::RecordEvent op_type_record_event("fake_dequantize_max_abs pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}}};
    
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

static PyObject * imperative_qr(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "qr";
    platform::RecordEvent op_type_record_event("qr pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Q", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"R", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Q"][0],outs["R"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_anchor_generator(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "anchor_generator";
    platform::RecordEvent op_type_record_event("anchor_generator pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Anchors", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Variances", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Anchors"][0],outs["Variances"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_layer_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "layer_norm";
    platform::RecordEvent op_type_record_event("layer_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, true);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Mean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Variance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Scale != nullptr) {
      ins["Scale"] = {Scale};
    }

    if (Bias != nullptr) {
      ins["Bias"] = {Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["Mean"][0],outs["Variance"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_merge_selected_rows(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "merge_selected_rows";
    platform::RecordEvent op_type_record_event("merge_selected_rows pybind_imperative_func");
    
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

static PyObject * imperative_acosh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "acosh";
    platform::RecordEvent op_type_record_event("acosh pybind_imperative_func");
    
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

static PyObject * imperative_stft(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "stft";
    platform::RecordEvent op_type_record_event("stft pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Window = GetVarBaseFromArgs(op_type, "Window", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Window", {Window}}};
    
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

static PyObject * imperative_less_equal(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "less_equal";
    platform::RecordEvent op_type_record_event("less_equal pybind_imperative_func");
    
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

static PyObject * imperative_rnn(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "rnn";
    platform::RecordEvent op_type_record_event("rnn pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto PreState = GetVarBaseListFromArgs(op_type, "PreState", args, 1, false);
    auto WeightList = GetVarBaseListFromArgs(op_type, "WeightList", args, 2, false);
    auto SequenceLength = GetVarBaseFromArgs(op_type, "SequenceLength", args, 3, true);
    auto DropoutState = GetVarBaseFromArgs(op_type, "DropoutState", args, 4, true);
    auto StateNum = GetUnsignedLongFromArgs(op_type, "StateNum", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Reserve", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"State", ConstructDuplicableOutput(StateNum)}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"PreState", PreState},{"WeightList", WeightList}};
    
    if (SequenceLength != nullptr) {
      ins["SequenceLength"] = {SequenceLength};
    }

    outs["DropoutState"] = {DropoutState};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["DropoutState"][0],outs["Reserve"][0],outs["Out"][0],outs["State"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_lstm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_lstm";
    platform::RecordEvent op_type_record_event("fusion_lstm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto WeightX = GetVarBaseFromArgs(op_type, "WeightX", args, 1, false);
    auto WeightH = GetVarBaseFromArgs(op_type, "WeightH", args, 2, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Hidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Cell", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"XX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchedInput", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchedHidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchedCell", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReorderedH0", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReorderedC0", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"CheckedCell", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"WeightX", {WeightX}},{"WeightH", {WeightH}},{"Bias", {Bias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Hidden"][0],outs["Cell"][0],outs["XX"][0],outs["BatchedInput"][0],outs["BatchedHidden"][0],outs["BatchedCell"][0],outs["ReorderedH0"][0],outs["ReorderedC0"][0],outs["CheckedCell"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lars_momentum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lars_momentum";
    platform::RecordEvent op_type_record_event("lars_momentum pybind_imperative_func");
    
    auto Param = GetVarBaseListFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseListFromArgs(op_type, "Grad", args, 1, false);
    auto Velocity = GetVarBaseListFromArgs(op_type, "Velocity", args, 2, false);
    auto LearningRate = GetVarBaseListFromArgs(op_type, "LearningRate", args, 3, false);
    auto ParamOut = GetVarBaseListFromArgs(op_type, "ParamOut", args, 4, false);
    auto VelocityOut = GetVarBaseListFromArgs(op_type, "VelocityOut", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", ParamOut},{"VelocityOut", VelocityOut}};
    imperative::NameVarBaseMap ins = {{"Param", Param},{"Grad", Grad},{"Velocity", Velocity},{"LearningRate", LearningRate}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"],outs["VelocityOut"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_hard_sigmoid(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "hard_sigmoid";
    platform::RecordEvent op_type_record_event("hard_sigmoid pybind_imperative_func");
    
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

static PyObject * imperative_hard_sigmoid_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "hard_sigmoid";
    platform::RecordEvent op_type_record_event("hard_sigmoid pybind_imperative_func");
    
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

static PyObject * imperative_isnan(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "isnan";
    platform::RecordEvent op_type_record_event("isnan pybind_imperative_func");
    
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

static PyObject * imperative_elementwise_floordiv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_floordiv";
    platform::RecordEvent op_type_record_event("elementwise_floordiv pybind_imperative_func");
    
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

static PyObject * imperative_correlation(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "correlation";
    platform::RecordEvent op_type_record_event("correlation pybind_imperative_func");
    
    auto Input1 = GetVarBaseFromArgs(op_type, "Input1", args, 0, false);
    auto Input2 = GetVarBaseFromArgs(op_type, "Input2", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input1", {Input1}},{"Input2", {Input2}}};
    
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

static PyObject * imperative_histogram(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "histogram";
    platform::RecordEvent op_type_record_event("histogram pybind_imperative_func");
    
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

static PyObject * imperative_gather_tree(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gather_tree";
    platform::RecordEvent op_type_record_event("gather_tree pybind_imperative_func");
    
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 0, false);
    auto Parents = GetVarBaseFromArgs(op_type, "Parents", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Ids", {Ids}},{"Parents", {Parents}}};
    
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

static PyObject * imperative_nanmedian(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "nanmedian";
    platform::RecordEvent op_type_record_event("nanmedian pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"MedianIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["MedianIndex"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_segment_pool(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "segment_pool";
    platform::RecordEvent op_type_record_event("segment_pool pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto SegmentIds = GetVarBaseFromArgs(op_type, "SegmentIds", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SummedIds", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"SegmentIds", {SegmentIds}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["SummedIds"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_repeated_fc_relu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_repeated_fc_relu";
    platform::RecordEvent op_type_record_event("fusion_repeated_fc_relu pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto W = GetVarBaseListFromArgs(op_type, "W", args, 1, false);
    auto Bias = GetVarBaseListFromArgs(op_type, "Bias", args, 2, false);
    auto ReluOutNum = GetUnsignedLongFromArgs(op_type, "ReluOutNum", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ReluOut", ConstructDuplicableOutput(ReluOutNum)},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"W", W},{"Bias", Bias}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ReluOut"],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sync_batch_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sync_batch_norm";
    platform::RecordEvent op_type_record_event("sync_batch_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    auto Mean = GetVarBaseFromArgs(op_type, "Mean", args, 3, false);
    auto Variance = GetVarBaseFromArgs(op_type, "Variance", args, 4, false);
    auto MeanOut = GetVarBaseFromArgs(op_type, "MeanOut", args, 5, false);
    auto VarianceOut = GetVarBaseFromArgs(op_type, "VarianceOut", args, 6, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 7, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MeanOut", {MeanOut}},{"VarianceOut", {VarianceOut}},{"SavedMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReserveSpace", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}},{"Bias", {Bias}},{"Mean", {Mean}},{"Variance", {Variance}}};
    
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

static PyObject * imperative_nop(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "nop";
    platform::RecordEvent op_type_record_event("nop pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
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

static PyObject * imperative_fused_attention(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_attention";
    platform::RecordEvent op_type_record_event("fused_attention pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto LnScale = GetVarBaseFromArgs(op_type, "LnScale", args, 1, true);
    auto LnBias = GetVarBaseFromArgs(op_type, "LnBias", args, 2, true);
    auto QKVW = GetVarBaseFromArgs(op_type, "QKVW", args, 3, false);
    auto QKVBias = GetVarBaseFromArgs(op_type, "QKVBias", args, 4, true);
    auto CacheKV = GetVarBaseFromArgs(op_type, "CacheKV", args, 5, true);
    auto SrcMask = GetVarBaseFromArgs(op_type, "SrcMask", args, 6, true);
    auto OutLinearW = GetVarBaseFromArgs(op_type, "OutLinearW", args, 7, false);
    auto OutLinearBias = GetVarBaseFromArgs(op_type, "OutLinearBias", args, 8, true);
    auto Ln2Scale = GetVarBaseFromArgs(op_type, "Ln2Scale", args, 9, true);
    auto Ln2Bias = GetVarBaseFromArgs(op_type, "Ln2Bias", args, 10, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 11, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"LnMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LnVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LnOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"QKVOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"QKVBiasOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TransposeOut2", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"QKOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"QKTVOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SoftmaxOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AttnDropoutMaskOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AttnDropoutOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SrcMaskOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"FMHAOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutLinearOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"DropoutMaskOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln2Mean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Ln2Variance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BiasDropoutResidualOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"CacheKVOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"QKVW", {QKVW}},{"OutLinearW", {OutLinearW}}};
    
    if (LnScale != nullptr) {
      ins["LnScale"] = {LnScale};
    }

    if (LnBias != nullptr) {
      ins["LnBias"] = {LnBias};
    }

    if (QKVBias != nullptr) {
      ins["QKVBias"] = {QKVBias};
    }

    if (CacheKV != nullptr) {
      ins["CacheKV"] = {CacheKV};
    }

    if (SrcMask != nullptr) {
      ins["SrcMask"] = {SrcMask};
    }

    if (OutLinearBias != nullptr) {
      ins["OutLinearBias"] = {OutLinearBias};
    }

    if (Ln2Scale != nullptr) {
      ins["Ln2Scale"] = {Ln2Scale};
    }

    if (Ln2Bias != nullptr) {
      ins["Ln2Bias"] = {Ln2Bias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["LnMean"][0],outs["LnVariance"][0],outs["LnOut"][0],outs["QKVOut"][0],outs["QKVBiasOut"][0],outs["TransposeOut2"][0],outs["QKOut"][0],outs["QKTVOut"][0],outs["SoftmaxOut"][0],outs["AttnDropoutMaskOut"][0],outs["AttnDropoutOut"][0],outs["SrcMaskOut"][0],outs["FMHAOut"][0],outs["OutLinearOut"][0],outs["DropoutMaskOut"][0],outs["Ln2Mean"][0],outs["Ln2Variance"][0],outs["BiasDropoutResidualOut"][0],outs["CacheKVOut"][0],outs["Y"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_filter_by_instag(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "filter_by_instag";
    platform::RecordEvent op_type_record_event("filter_by_instag pybind_imperative_func");
    
    auto Ins = GetVarBaseFromArgs(op_type, "Ins", args, 0, false);
    auto Ins_tag = GetVarBaseFromArgs(op_type, "Ins_tag", args, 1, false);
    auto Filter_tag = GetVarBaseFromArgs(op_type, "Filter_tag", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LossWeight", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"IndexMap", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Ins", {Ins}},{"Ins_tag", {Ins_tag}},{"Filter_tag", {Filter_tag}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["LossWeight"][0],outs["IndexMap"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_expand_as_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "expand_as_v2";
    platform::RecordEvent op_type_record_event("expand_as_v2 pybind_imperative_func");
    
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

static PyObject * imperative_diag_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "diag_v2";
    platform::RecordEvent op_type_record_event("diag_v2 pybind_imperative_func");
    
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

static PyObject * imperative_pull_box_sparse(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pull_box_sparse";
    platform::RecordEvent op_type_record_event("pull_box_sparse pybind_imperative_func");
    
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

static PyObject * imperative_nll_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "nll_loss";
    platform::RecordEvent op_type_record_event("nll_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Total_weight", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Label", {Label}}};
    
    if (Weight != nullptr) {
      ins["Weight"] = {Weight};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Total_weight"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sparse_relu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sparse_relu";
    platform::RecordEvent op_type_record_event("sparse_relu pybind_imperative_func");
    
    auto x = GetVarBaseFromArgs(op_type, "x", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"x", {x}}};
    
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

static PyObject * imperative_dot(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dot";
    platform::RecordEvent op_type_record_event("dot pybind_imperative_func");
    
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

static PyObject * imperative_fused_token_prune(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_token_prune";
    platform::RecordEvent op_type_record_event("fused_token_prune pybind_imperative_func");
    
    auto Attn = GetVarBaseFromArgs(op_type, "Attn", args, 0, false);
    auto X = GetVarBaseFromArgs(op_type, "X", args, 1, false);
    auto Mask = GetVarBaseFromArgs(op_type, "Mask", args, 2, false);
    auto NewMask = GetVarBaseFromArgs(op_type, "NewMask", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"SlimmedX", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"CLSInds", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Attn", {Attn}},{"X", {X}},{"Mask", {Mask}},{"NewMask", {NewMask}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["SlimmedX"][0],outs["CLSInds"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_scale(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "scale";
    platform::RecordEvent op_type_record_event("scale pybind_imperative_func");
    
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

static PyObject * imperative_scale_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "scale";
    platform::RecordEvent op_type_record_event("scale pybind_imperative_func");
    
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

static PyObject * imperative_shuffle_batch(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "shuffle_batch";
    platform::RecordEvent op_type_record_event("shuffle_batch pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Seed = GetVarBaseFromArgs(op_type, "Seed", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ShuffleIdx", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SeedOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Seed", {Seed}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["ShuffleIdx"][0],outs["SeedOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_diag(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "diag";
    platform::RecordEvent op_type_record_event("diag pybind_imperative_func");
    
    auto Diagonal = GetVarBaseFromArgs(op_type, "Diagonal", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Diagonal", {Diagonal}}};
    
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

static PyObject * imperative_multiplex(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "multiplex";
    platform::RecordEvent op_type_record_event("multiplex pybind_imperative_func");
    
    auto Ids = GetVarBaseFromArgs(op_type, "Ids", args, 0, false);
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Ids", {Ids}},{"X", X}};
    
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

static PyObject * imperative_leaky_relu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "leaky_relu";
    platform::RecordEvent op_type_record_event("leaky_relu pybind_imperative_func");
    
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

static PyObject * imperative_leaky_relu_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "leaky_relu";
    platform::RecordEvent op_type_record_event("leaky_relu pybind_imperative_func");
    
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

static PyObject * imperative_allclose(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "allclose";
    platform::RecordEvent op_type_record_event("allclose pybind_imperative_func");
    
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

static PyObject * imperative_adamw(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "adamw";
    platform::RecordEvent op_type_record_event("adamw pybind_imperative_func");
    
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
    auto Beta1PowOut = GetVarBaseFromArgs(op_type, "Beta1PowOut", args, 11, false);
    auto Beta2PowOut = GetVarBaseFromArgs(op_type, "Beta2PowOut", args, 12, false);
    auto MasterParamOut = GetVarBaseFromArgs(op_type, "MasterParamOut", args, 13, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 14, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}},{"Moment1Out", {Moment1Out}},{"Moment2Out", {Moment2Out}},{"Beta1PowOut", {Beta1PowOut}},{"Beta2PowOut", {Beta2PowOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"Grad", {Grad}},{"LearningRate", {LearningRate}},{"Moment1", {Moment1}},{"Moment2", {Moment2}},{"Beta1Pow", {Beta1Pow}},{"Beta2Pow", {Beta2Pow}}};
    
    if (MasterParam != nullptr) {
      ins["MasterParam"] = {MasterParam};
    }

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

static PyObject * imperative_elementwise_pow(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_pow";
    platform::RecordEvent op_type_record_event("elementwise_pow pybind_imperative_func");
    
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

static PyObject * imperative_prior_box(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "prior_box";
    platform::RecordEvent op_type_record_event("prior_box pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Image = GetVarBaseFromArgs(op_type, "Image", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Boxes", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Variances", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Image", {Image}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Boxes"][0],outs["Variances"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_p_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "p_norm";
    platform::RecordEvent op_type_record_event("p_norm pybind_imperative_func");
    
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

static PyObject * imperative_fused_gate_attention(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_gate_attention";
    platform::RecordEvent op_type_record_event("fused_gate_attention pybind_imperative_func");
    
    auto Query = GetVarBaseFromArgs(op_type, "Query", args, 0, false);
    auto Key = GetVarBaseFromArgs(op_type, "Key", args, 1, true);
    auto QueryWeight = GetVarBaseFromArgs(op_type, "QueryWeight", args, 2, true);
    auto KeyWeight = GetVarBaseFromArgs(op_type, "KeyWeight", args, 3, true);
    auto ValueWeight = GetVarBaseFromArgs(op_type, "ValueWeight", args, 4, true);
    auto QKVWeight = GetVarBaseFromArgs(op_type, "QKVWeight", args, 5, true);
    auto NonbatchedBias = GetVarBaseFromArgs(op_type, "NonbatchedBias", args, 6, true);
    auto SrcMask = GetVarBaseFromArgs(op_type, "SrcMask", args, 7, false);
    auto GateWeight = GetVarBaseFromArgs(op_type, "GateWeight", args, 8, true);
    auto GateBias = GetVarBaseFromArgs(op_type, "GateBias", args, 9, true);
    auto OutLinearWeight = GetVarBaseFromArgs(op_type, "OutLinearWeight", args, 10, false);
    auto OutLinearBias = GetVarBaseFromArgs(op_type, "OutLinearBias", args, 11, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 12, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"QueryTransposeOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"KeyTransposeOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ValueTransposeOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"QKVTransposeOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SoftmaxOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"FMHAOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"GateOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Query", {Query}},{"SrcMask", {SrcMask}},{"OutLinearWeight", {OutLinearWeight}},{"OutLinearBias", {OutLinearBias}}};
    
    if (Key != nullptr) {
      ins["Key"] = {Key};
    }

    if (QueryWeight != nullptr) {
      ins["QueryWeight"] = {QueryWeight};
    }

    if (KeyWeight != nullptr) {
      ins["KeyWeight"] = {KeyWeight};
    }

    if (ValueWeight != nullptr) {
      ins["ValueWeight"] = {ValueWeight};
    }

    if (QKVWeight != nullptr) {
      ins["QKVWeight"] = {QKVWeight};
    }

    if (NonbatchedBias != nullptr) {
      ins["NonbatchedBias"] = {NonbatchedBias};
    }

    if (GateWeight != nullptr) {
      ins["GateWeight"] = {GateWeight};
    }

    if (GateBias != nullptr) {
      ins["GateBias"] = {GateBias};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["QueryTransposeOut"][0],outs["KeyTransposeOut"][0],outs["ValueTransposeOut"][0],outs["QKVTransposeOut"][0],outs["SoftmaxOut"][0],outs["FMHAOut"][0],outs["GateOut"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_unique_consecutive(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unique_consecutive";
    platform::RecordEvent op_type_record_event("unique_consecutive pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Index", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Counts", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Index"][0],outs["Counts"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lod_reset(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lod_reset";
    platform::RecordEvent op_type_record_event("lod_reset pybind_imperative_func");
    
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

static PyObject * imperative_lod_reset_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lod_reset";
    platform::RecordEvent op_type_record_event("lod_reset pybind_imperative_func");
    
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

static PyMethodDef ExtestMethods[] = {
  {"dirichlet", (PyCFunction)(void(*)(void))imperative_dirichlet, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dirichlet in dygraph."},
  {"stanh", (PyCFunction)(void(*)(void))imperative_stanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stanh in dygraph."},
  {"label_smooth", (PyCFunction)(void(*)(void))imperative_label_smooth, METH_VARARGS | METH_KEYWORDS, "C++ interface function for label_smooth in dygraph."},
  {"fold", (PyCFunction)(void(*)(void))imperative_fold, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fold in dygraph."},
  {"merged_momentum", (PyCFunction)(void(*)(void))imperative_merged_momentum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for merged_momentum in dygraph."},
  {"ascend_trigger", (PyCFunction)(void(*)(void))imperative_ascend_trigger, METH_VARARGS | METH_KEYWORDS, "C++ interface function for ascend_trigger in dygraph."},
  {"rpn_target_assign", (PyCFunction)(void(*)(void))imperative_rpn_target_assign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rpn_target_assign in dygraph."},
  {"fused_feedforward", (PyCFunction)(void(*)(void))imperative_fused_feedforward, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_feedforward in dygraph."},
  {"roi_perspective_transform", (PyCFunction)(void(*)(void))imperative_roi_perspective_transform, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_perspective_transform in dygraph."},
  {"expand", (PyCFunction)(void(*)(void))imperative_expand, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expand in dygraph."},
  {"prroi_pool", (PyCFunction)(void(*)(void))imperative_prroi_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prroi_pool in dygraph."},
  {"pool3d", (PyCFunction)(void(*)(void))imperative_pool3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pool3d in dygraph."},
  {"memcpy", (PyCFunction)(void(*)(void))imperative_memcpy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memcpy in dygraph."},
  {"distribute_fpn_proposals", (PyCFunction)(void(*)(void))imperative_distribute_fpn_proposals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distribute_fpn_proposals in dygraph."},
  {"frame", (PyCFunction)(void(*)(void))imperative_frame, METH_VARARGS | METH_KEYWORDS, "C++ interface function for frame in dygraph."},
  {"bincount", (PyCFunction)(void(*)(void))imperative_bincount, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bincount in dygraph."},
  {"shape", (PyCFunction)(void(*)(void))imperative_shape, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shape in dygraph."},
  {"mode", (PyCFunction)(void(*)(void))imperative_mode, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mode in dygraph."},
  {"group_norm", (PyCFunction)(void(*)(void))imperative_group_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for group_norm in dygraph."},
  {"resnet_unit", (PyCFunction)(void(*)(void))imperative_resnet_unit, METH_VARARGS | METH_KEYWORDS, "C++ interface function for resnet_unit in dygraph."},
  {"sequence_expand_as", (PyCFunction)(void(*)(void))imperative_sequence_expand_as, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_expand_as in dygraph."},
  {"cos_sim", (PyCFunction)(void(*)(void))imperative_cos_sim, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cos_sim in dygraph."},
  {"eigvals", (PyCFunction)(void(*)(void))imperative_eigvals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for eigvals in dygraph."},
  {"save_combine", (PyCFunction)(void(*)(void))imperative_save_combine, METH_VARARGS | METH_KEYWORDS, "C++ interface function for save_combine in dygraph."},
  {"class_center_sample", (PyCFunction)(void(*)(void))imperative_class_center_sample, METH_VARARGS | METH_KEYWORDS, "C++ interface function for class_center_sample in dygraph."},
  {"elementwise_fmin", (PyCFunction)(void(*)(void))imperative_elementwise_fmin, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_fmin in dygraph."},
  {"read_file", (PyCFunction)(void(*)(void))imperative_read_file, METH_VARARGS | METH_KEYWORDS, "C++ interface function for read_file in dygraph."},
  {"isfinite", (PyCFunction)(void(*)(void))imperative_isfinite, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isfinite in dygraph."},
  {"arg_max", (PyCFunction)(void(*)(void))imperative_arg_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for arg_max in dygraph."},
  {"equal", (PyCFunction)(void(*)(void))imperative_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for equal in dygraph."},
  {"fake_dequantize_max_abs", (PyCFunction)(void(*)(void))imperative_fake_dequantize_max_abs, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_dequantize_max_abs in dygraph."},
  {"qr", (PyCFunction)(void(*)(void))imperative_qr, METH_VARARGS | METH_KEYWORDS, "C++ interface function for qr in dygraph."},
  {"anchor_generator", (PyCFunction)(void(*)(void))imperative_anchor_generator, METH_VARARGS | METH_KEYWORDS, "C++ interface function for anchor_generator in dygraph."},
  {"layer_norm", (PyCFunction)(void(*)(void))imperative_layer_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for layer_norm in dygraph."},
  {"merge_selected_rows", (PyCFunction)(void(*)(void))imperative_merge_selected_rows, METH_VARARGS | METH_KEYWORDS, "C++ interface function for merge_selected_rows in dygraph."},
  {"acosh", (PyCFunction)(void(*)(void))imperative_acosh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acosh in dygraph."},
  {"stft", (PyCFunction)(void(*)(void))imperative_stft, METH_VARARGS | METH_KEYWORDS, "C++ interface function for stft in dygraph."},
  {"less_equal", (PyCFunction)(void(*)(void))imperative_less_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for less_equal in dygraph."},
  {"rnn", (PyCFunction)(void(*)(void))imperative_rnn, METH_VARARGS | METH_KEYWORDS, "C++ interface function for rnn in dygraph."},
  {"fusion_lstm", (PyCFunction)(void(*)(void))imperative_fusion_lstm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_lstm in dygraph."},
  {"lars_momentum", (PyCFunction)(void(*)(void))imperative_lars_momentum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lars_momentum in dygraph."},
  {"hard_sigmoid", (PyCFunction)(void(*)(void))imperative_hard_sigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hard_sigmoid in dygraph."},
  {"hard_sigmoid_", (PyCFunction)(void(*)(void))imperative_hard_sigmoid_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hard_sigmoid_ in dygraph."},
  {"isnan", (PyCFunction)(void(*)(void))imperative_isnan, METH_VARARGS | METH_KEYWORDS, "C++ interface function for isnan in dygraph."},
  {"elementwise_floordiv", (PyCFunction)(void(*)(void))imperative_elementwise_floordiv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_floordiv in dygraph."},
  {"correlation", (PyCFunction)(void(*)(void))imperative_correlation, METH_VARARGS | METH_KEYWORDS, "C++ interface function for correlation in dygraph."},
  {"histogram", (PyCFunction)(void(*)(void))imperative_histogram, METH_VARARGS | METH_KEYWORDS, "C++ interface function for histogram in dygraph."},
  {"gather_tree", (PyCFunction)(void(*)(void))imperative_gather_tree, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gather_tree in dygraph."},
  {"nanmedian", (PyCFunction)(void(*)(void))imperative_nanmedian, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nanmedian in dygraph."},
  {"segment_pool", (PyCFunction)(void(*)(void))imperative_segment_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for segment_pool in dygraph."},
  {"fusion_repeated_fc_relu", (PyCFunction)(void(*)(void))imperative_fusion_repeated_fc_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_repeated_fc_relu in dygraph."},
  {"sync_batch_norm", (PyCFunction)(void(*)(void))imperative_sync_batch_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sync_batch_norm in dygraph."},
  {"nop", (PyCFunction)(void(*)(void))imperative_nop, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nop in dygraph."},
  {"fused_attention", (PyCFunction)(void(*)(void))imperative_fused_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_attention in dygraph."},
  {"filter_by_instag", (PyCFunction)(void(*)(void))imperative_filter_by_instag, METH_VARARGS | METH_KEYWORDS, "C++ interface function for filter_by_instag in dygraph."},
  {"expand_as_v2", (PyCFunction)(void(*)(void))imperative_expand_as_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expand_as_v2 in dygraph."},
  {"diag_v2", (PyCFunction)(void(*)(void))imperative_diag_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diag_v2 in dygraph."},
  {"pull_box_sparse", (PyCFunction)(void(*)(void))imperative_pull_box_sparse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pull_box_sparse in dygraph."},
  {"nll_loss", (PyCFunction)(void(*)(void))imperative_nll_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nll_loss in dygraph."},
  {"sparse_relu", (PyCFunction)(void(*)(void))imperative_sparse_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_relu in dygraph."},
  {"dot", (PyCFunction)(void(*)(void))imperative_dot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dot in dygraph."},
  {"fused_token_prune", (PyCFunction)(void(*)(void))imperative_fused_token_prune, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_token_prune in dygraph."},
  {"scale", (PyCFunction)(void(*)(void))imperative_scale, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scale in dygraph."},
  {"scale_", (PyCFunction)(void(*)(void))imperative_scale_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for scale_ in dygraph."},
  {"shuffle_batch", (PyCFunction)(void(*)(void))imperative_shuffle_batch, METH_VARARGS | METH_KEYWORDS, "C++ interface function for shuffle_batch in dygraph."},
  {"diag", (PyCFunction)(void(*)(void))imperative_diag, METH_VARARGS | METH_KEYWORDS, "C++ interface function for diag in dygraph."},
  {"multiplex", (PyCFunction)(void(*)(void))imperative_multiplex, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multiplex in dygraph."},
  {"leaky_relu", (PyCFunction)(void(*)(void))imperative_leaky_relu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for leaky_relu in dygraph."},
  {"leaky_relu_", (PyCFunction)(void(*)(void))imperative_leaky_relu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for leaky_relu_ in dygraph."},
  {"allclose", (PyCFunction)(void(*)(void))imperative_allclose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for allclose in dygraph."},
  {"adamw", (PyCFunction)(void(*)(void))imperative_adamw, METH_VARARGS | METH_KEYWORDS, "C++ interface function for adamw in dygraph."},
  {"elementwise_pow", (PyCFunction)(void(*)(void))imperative_elementwise_pow, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_pow in dygraph."},
  {"prior_box", (PyCFunction)(void(*)(void))imperative_prior_box, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prior_box in dygraph."},
  {"p_norm", (PyCFunction)(void(*)(void))imperative_p_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for p_norm in dygraph."},
  {"fused_gate_attention", (PyCFunction)(void(*)(void))imperative_fused_gate_attention, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_gate_attention in dygraph."},
  {"unique_consecutive", (PyCFunction)(void(*)(void))imperative_unique_consecutive, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unique_consecutive in dygraph."},
  {"lod_reset", (PyCFunction)(void(*)(void))imperative_lod_reset, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lod_reset in dygraph."},
  {"lod_reset_", (PyCFunction)(void(*)(void))imperative_lod_reset_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lod_reset_ in dygraph."},
  {nullptr,nullptr,0,nullptr}};

void BindOpFunctions5(pybind11::module *module) {
  auto m = module->def_submodule("ops");
  if (PyModule_AddFunctions(m.ptr(), ExtestMethods) < 0) {
    PADDLE_THROW(platform::errors::Fatal ("Add functions to core.ops failed!"));
  }

  InitOpsAttrTypeMap();}

} // namespace pybind
} // namespace paddle
