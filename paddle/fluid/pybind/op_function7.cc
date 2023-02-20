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

static PyObject * imperative_fused_embedding_seq_pool(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_embedding_seq_pool";
    platform::RecordEvent op_type_record_event("fused_embedding_seq_pool pybind_imperative_func");
    
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

static PyObject * imperative_kthvalue(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "kthvalue";
    platform::RecordEvent op_type_record_event("kthvalue pybind_imperative_func");
    
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

static PyObject * imperative_graph_send_uv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "graph_send_uv";
    platform::RecordEvent op_type_record_event("graph_send_uv pybind_imperative_func");
    
    auto x = GetVarBaseFromArgs(op_type, "x", args, 0, false);
    auto y = GetVarBaseFromArgs(op_type, "y", args, 1, false);
    auto src_index = GetVarBaseFromArgs(op_type, "src_index", args, 2, false);
    auto dst_index = GetVarBaseFromArgs(op_type, "dst_index", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"x", {x}},{"y", {y}},{"src_index", {src_index}},{"dst_index", {dst_index}}};
    
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

static PyObject * imperative_erf(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "erf";
    platform::RecordEvent op_type_record_event("erf pybind_imperative_func");
    
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

static PyObject * imperative_yolo_box_post(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "yolo_box_post";
    platform::RecordEvent op_type_record_event("yolo_box_post pybind_imperative_func");
    
    auto Boxes0 = GetVarBaseFromArgs(op_type, "Boxes0", args, 0, false);
    auto Boxes1 = GetVarBaseFromArgs(op_type, "Boxes1", args, 1, false);
    auto Boxes2 = GetVarBaseFromArgs(op_type, "Boxes2", args, 2, false);
    auto ImageShape = GetVarBaseFromArgs(op_type, "ImageShape", args, 3, false);
    auto ImageScale = GetVarBaseFromArgs(op_type, "ImageScale", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"NmsRoisNum", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Boxes0", {Boxes0}},{"Boxes1", {Boxes1}},{"Boxes2", {Boxes2}},{"ImageShape", {ImageShape}},{"ImageScale", {ImageScale}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["NmsRoisNum"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_conv2d_inception_fusion(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conv2d_inception_fusion";
    platform::RecordEvent op_type_record_event("conv2d_inception_fusion pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Filter = GetVarBaseListFromArgs(op_type, "Filter", args, 1, false);
    auto Bias = GetVarBaseListFromArgs(op_type, "Bias", args, 2, false);
    auto TempOutputNum = GetUnsignedLongFromArgs(op_type, "TempOutputNum", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"TempOutput", ConstructDuplicableOutput(TempOutputNum)}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Filter", Filter},{"Bias", Bias}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Output"][0],outs["TempOutput"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_logsumexp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logsumexp";
    platform::RecordEvent op_type_record_event("logsumexp pybind_imperative_func");
    
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

static PyObject * imperative_trilinear_interp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "trilinear_interp";
    platform::RecordEvent op_type_record_event("trilinear_interp pybind_imperative_func");
    
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

static PyObject * imperative_fusion_seqpool_concat(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_seqpool_concat";
    platform::RecordEvent op_type_record_event("fusion_seqpool_concat pybind_imperative_func");
    
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

static PyObject * imperative_alloc_float_status(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "alloc_float_status";
    platform::RecordEvent op_type_record_event("alloc_float_status pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"FloatStatus", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["FloatStatus"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sequence_concat(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_concat";
    platform::RecordEvent op_type_record_event("sequence_concat pybind_imperative_func");
    
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

static PyObject * imperative_fusion_seqpool_cvm_concat(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_seqpool_cvm_concat";
    platform::RecordEvent op_type_record_event("fusion_seqpool_cvm_concat pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto CVM = GetVarBaseFromArgs(op_type, "CVM", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X},{"CVM", {CVM}}};
    
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

static PyObject * imperative_unpool3d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unpool3d";
    platform::RecordEvent op_type_record_event("unpool3d pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Indices = GetVarBaseFromArgs(op_type, "Indices", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Indices", {Indices}}};
    
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

static PyObject * imperative_similarity_focus(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "similarity_focus";
    platform::RecordEvent op_type_record_event("similarity_focus pybind_imperative_func");
    
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

static PyObject * imperative_argsort(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "argsort";
    platform::RecordEvent op_type_record_event("argsort pybind_imperative_func");
    
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

static PyObject * imperative_sequence_expand(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_expand";
    platform::RecordEvent op_type_record_event("sequence_expand pybind_imperative_func");
    
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

static PyObject * imperative_fused_bn_add_activation(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_bn_add_activation";
    platform::RecordEvent op_type_record_event("fused_bn_add_activation pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Z = GetVarBaseFromArgs(op_type, "Z", args, 1, false);
    auto Scale = GetVarBaseFromArgs(op_type, "Scale", args, 2, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MeanOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"VarianceOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReserveSpace", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Z", {Z}},{"Scale", {Scale}},{"Bias", {Bias}}};
    
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

static PyObject * imperative_sgd(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sgd";
    platform::RecordEvent op_type_record_event("sgd pybind_imperative_func");
    
    auto Param = GetVarBaseFromArgs(op_type, "Param", args, 0, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 1, false);
    auto Grad = GetVarBaseFromArgs(op_type, "Grad", args, 2, false);
    auto MasterParam = GetVarBaseFromArgs(op_type, "MasterParam", args, 3, true);
    auto ParamOut = GetVarBaseFromArgs(op_type, "ParamOut", args, 4, false);
    auto MasterParamOut = GetVarBaseFromArgs(op_type, "MasterParamOut", args, 5, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ParamOut", {ParamOut}}};
    imperative::NameVarBaseMap ins = {{"Param", {Param}},{"LearningRate", {LearningRate}},{"Grad", {Grad}}};
    
    if (MasterParam != nullptr) {
      ins["MasterParam"] = {MasterParam};
    }

    outs["MasterParamOut"] = {MasterParamOut};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ParamOut"][0],outs["MasterParamOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_exponential(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "exponential";
    platform::RecordEvent op_type_record_event("exponential pybind_imperative_func");
    
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

static PyObject * imperative_exponential_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "exponential";
    platform::RecordEvent op_type_record_event("exponential pybind_imperative_func");
    
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

static PyObject * imperative_bilinear_interp_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bilinear_interp_v2";
    platform::RecordEvent op_type_record_event("bilinear_interp_v2 pybind_imperative_func");
    
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

static PyObject * imperative_atanh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "atanh";
    platform::RecordEvent op_type_record_event("atanh pybind_imperative_func");
    
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

static PyObject * imperative_clip(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "clip";
    platform::RecordEvent op_type_record_event("clip pybind_imperative_func");
    
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

static PyObject * imperative_clip_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "clip";
    platform::RecordEvent op_type_record_event("clip pybind_imperative_func");
    
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

static PyObject * imperative_sparse_to_dense(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sparse_to_dense";
    platform::RecordEvent op_type_record_event("sparse_to_dense pybind_imperative_func");
    
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

static PyObject * imperative_deformable_conv_v1(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "deformable_conv_v1";
    platform::RecordEvent op_type_record_event("deformable_conv_v1 pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Offset = GetVarBaseFromArgs(op_type, "Offset", args, 1, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Offset", {Offset}},{"Filter", {Filter}}};
    
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

static PyObject * imperative_hinge_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "hinge_loss";
    platform::RecordEvent op_type_record_event("hinge_loss pybind_imperative_func");
    
    auto Logits = GetVarBaseFromArgs(op_type, "Logits", args, 0, false);
    auto Labels = GetVarBaseFromArgs(op_type, "Labels", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Logits", {Logits}},{"Labels", {Labels}}};
    
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

static PyObject * imperative_determinant(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "determinant";
    platform::RecordEvent op_type_record_event("determinant pybind_imperative_func");
    
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

static PyObject * imperative_conv2d_transpose(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conv2d_transpose";
    platform::RecordEvent op_type_record_event("conv2d_transpose pybind_imperative_func");
    
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

static PyObject * imperative_memcpy_d2h(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "memcpy_d2h";
    platform::RecordEvent op_type_record_event("memcpy_d2h pybind_imperative_func");
    
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

static PyObject * imperative_softsign(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "softsign";
    platform::RecordEvent op_type_record_event("softsign pybind_imperative_func");
    
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

static PyObject * imperative_fake_quantize_dequantize_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_quantize_dequantize_abs_max";
    platform::RecordEvent op_type_record_event("fake_quantize_dequantize_abs_max pybind_imperative_func");
    
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

static PyObject * imperative_broadcast_tensors(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "broadcast_tensors";
    platform::RecordEvent op_type_record_event("broadcast_tensors pybind_imperative_func");
    
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

static PyObject * imperative_cholesky_solve(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cholesky_solve";
    platform::RecordEvent op_type_record_event("cholesky_solve pybind_imperative_func");
    
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

static PyObject * imperative_grid_sampler(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "grid_sampler";
    platform::RecordEvent op_type_record_event("grid_sampler pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Grid = GetVarBaseFromArgs(op_type, "Grid", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Grid", {Grid}}};
    
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

static PyObject * imperative_pyramid_hash(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pyramid_hash";
    platform::RecordEvent op_type_record_event("pyramid_hash pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto W = GetVarBaseFromArgs(op_type, "W", args, 1, false);
    auto WhiteList = GetVarBaseFromArgs(op_type, "WhiteList", args, 2, false);
    auto BlackList = GetVarBaseFromArgs(op_type, "BlackList", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"DropPos", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"X_Temp_Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"W", {W}},{"WhiteList", {WhiteList}},{"BlackList", {BlackList}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["DropPos"][0],outs["X_Temp_Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fft_c2r(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fft_c2r";
    platform::RecordEvent op_type_record_event("fft_c2r pybind_imperative_func");
    
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

static PyObject * imperative_fake_quantize_dequantize_moving_average_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_quantize_dequantize_moving_average_abs_max";
    platform::RecordEvent op_type_record_event("fake_quantize_dequantize_moving_average_abs_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto InScale = GetVarBaseFromArgs(op_type, "InScale", args, 1, false);
    auto InAccum = GetVarBaseFromArgs(op_type, "InAccum", args, 2, true);
    auto InState = GetVarBaseFromArgs(op_type, "InState", args, 3, true);
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 4, false);
    auto OutScale = GetVarBaseFromArgs(op_type, "OutScale", args, 5, false);
    auto OutState = GetVarBaseFromArgs(op_type, "OutState", args, 6, true);
    auto OutAccum = GetVarBaseFromArgs(op_type, "OutAccum", args, 7, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 8, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}},{"OutScale", {OutScale}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"InScale", {InScale}}};
    
    if (InAccum != nullptr) {
      ins["InAccum"] = {InAccum};
    }

    if (InState != nullptr) {
      ins["InState"] = {InState};
    }

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

static PyObject * imperative_multi_dot(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "multi_dot";
    platform::RecordEvent op_type_record_event("multi_dot pybind_imperative_func");
    
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

static PyObject * imperative_sequence_pool(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_pool";
    platform::RecordEvent op_type_record_event("sequence_pool pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MaxIndex", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["MaxIndex"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_transpose(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "transpose";
    platform::RecordEvent op_type_record_event("transpose pybind_imperative_func");
    
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

static PyObject * imperative_top_k(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "top_k";
    platform::RecordEvent op_type_record_event("top_k pybind_imperative_func");
    
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

static PyObject * imperative_renorm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "renorm";
    platform::RecordEvent op_type_record_event("renorm pybind_imperative_func");
    
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

static PyObject * imperative_pixel_unshuffle(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pixel_unshuffle";
    platform::RecordEvent op_type_record_event("pixel_unshuffle pybind_imperative_func");
    
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

static PyObject * imperative_take_along_axis(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "take_along_axis";
    platform::RecordEvent op_type_record_event("take_along_axis pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Result", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Index", {Index}}};
    
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

static PyObject * imperative_dist(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dist";
    platform::RecordEvent op_type_record_event("dist pybind_imperative_func");
    
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

static PyObject * imperative_affine_grid(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "affine_grid";
    platform::RecordEvent op_type_record_event("affine_grid pybind_imperative_func");
    
    auto Theta = GetVarBaseFromArgs(op_type, "Theta", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Theta", {Theta}}};
    
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

static PyObject * imperative_gaussian_random_batch_size_like(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gaussian_random_batch_size_like";
    platform::RecordEvent op_type_record_event("gaussian_random_batch_size_like pybind_imperative_func");
    
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

static PyObject * imperative_fake_channel_wise_dequantize_max_abs(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_channel_wise_dequantize_max_abs";
    platform::RecordEvent op_type_record_event("fake_channel_wise_dequantize_max_abs pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Scales = GetVarBaseListFromArgs(op_type, "Scales", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scales", Scales}};
    
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

static PyObject * imperative_reciprocal(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reciprocal";
    platform::RecordEvent op_type_record_event("reciprocal pybind_imperative_func");
    
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

static PyObject * imperative_reciprocal_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "reciprocal";
    platform::RecordEvent op_type_record_event("reciprocal pybind_imperative_func");
    
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

static PyObject * imperative_sequence_mask(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_mask";
    platform::RecordEvent op_type_record_event("sequence_mask pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto MaxLenTensor = GetVarBaseFromArgs(op_type, "MaxLenTensor", args, 1, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (MaxLenTensor != nullptr) {
      ins["MaxLenTensor"] = {MaxLenTensor};
    }

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

static PyObject * imperative_prune_gate_by_capacity(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "prune_gate_by_capacity";
    platform::RecordEvent op_type_record_event("prune_gate_by_capacity pybind_imperative_func");
    
    auto GateIdx = GetVarBaseFromArgs(op_type, "GateIdx", args, 0, false);
    auto ExpertCount = GetVarBaseFromArgs(op_type, "ExpertCount", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"NewGateIdx", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"GateIdx", {GateIdx}},{"ExpertCount", {ExpertCount}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["NewGateIdx"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fill_diagonal_tensor(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_diagonal_tensor";
    platform::RecordEvent op_type_record_event("fill_diagonal_tensor pybind_imperative_func");
    
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

static PyObject * imperative_fill_diagonal_tensor_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_diagonal_tensor";
    platform::RecordEvent op_type_record_event("fill_diagonal_tensor pybind_imperative_func");
    
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

static PyObject * imperative_abs(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "abs";
    platform::RecordEvent op_type_record_event("abs pybind_imperative_func");
    
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

static PyObject * imperative_partial_concat(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "partial_concat";
    platform::RecordEvent op_type_record_event("partial_concat pybind_imperative_func");
    
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

static PyObject * imperative_elu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elu";
    platform::RecordEvent op_type_record_event("elu pybind_imperative_func");
    
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

static PyObject * imperative_elu_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elu";
    platform::RecordEvent op_type_record_event("elu pybind_imperative_func");
    
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

static PyObject * imperative_index_select(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "index_select";
    platform::RecordEvent op_type_record_event("index_select pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Index = GetVarBaseFromArgs(op_type, "Index", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Index", {Index}}};
    
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

static PyObject * imperative_row_conv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "row_conv";
    platform::RecordEvent op_type_record_event("row_conv pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Filter", {Filter}}};
    
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

static PyObject * imperative_cross(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cross";
    platform::RecordEvent op_type_record_event("cross pybind_imperative_func");
    
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

static PyObject * imperative_elementwise_mul(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_mul";
    platform::RecordEvent op_type_record_event("elementwise_mul pybind_imperative_func");
    
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

static PyObject * imperative_decayed_adagrad(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "decayed_adagrad";
    platform::RecordEvent op_type_record_event("decayed_adagrad pybind_imperative_func");
    
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

static PyObject * imperative_bipartite_match(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bipartite_match";
    platform::RecordEvent op_type_record_event("bipartite_match pybind_imperative_func");
    
    auto DistMat = GetVarBaseFromArgs(op_type, "DistMat", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"ColToRowMatchIndices", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ColToRowMatchDist", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"DistMat", {DistMat}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["ColToRowMatchIndices"][0],outs["ColToRowMatchDist"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_run_program(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "run_program";
    platform::RecordEvent op_type_record_event("run_program pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto Params = GetVarBaseListFromArgs(op_type, "Params", args, 1, true);
    auto Out = GetVarBaseListFromArgs(op_type, "Out", args, 2, false);
    auto OutScope = GetVarBaseFromArgs(op_type, "OutScope", args, 3, false);
    auto DOut = GetVarBaseListFromArgs(op_type, "DOut", args, 4, true);
    auto CUDAGraph = GetVarBaseFromArgs(op_type, "CUDAGraph", args, 5, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", Out},{"OutScope", {OutScope}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
    if (Params.size() != 0) {
      ins["Params"] = Params;
    }

    outs["DOut"] = DOut;

    outs["CUDAGraph"] = {CUDAGraph};

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"],outs["OutScope"][0],outs["DOut"],outs["CUDAGraph"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fake_quantize_moving_average_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_quantize_moving_average_abs_max";
    platform::RecordEvent op_type_record_event("fake_quantize_moving_average_abs_max pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto InScale = GetVarBaseFromArgs(op_type, "InScale", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutScale", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"InScale", {InScale}}};
    
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

static PyObject * imperative_fused_multi_transformer_int8(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_multi_transformer_int8";
    platform::RecordEvent op_type_record_event("fused_multi_transformer_int8 pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto LnScale = GetVarBaseListFromArgs(op_type, "LnScale", args, 1, false);
    auto LnBias = GetVarBaseListFromArgs(op_type, "LnBias", args, 2, false);
    auto QKVW = GetVarBaseListFromArgs(op_type, "QKVW", args, 3, false);
    auto QKVBias = GetVarBaseListFromArgs(op_type, "QKVBias", args, 4, true);
    auto CacheKV = GetVarBaseListFromArgs(op_type, "CacheKV", args, 5, true);
    auto TimeStep = GetVarBaseFromArgs(op_type, "TimeStep", args, 6, true);
    auto SrcMask = GetVarBaseFromArgs(op_type, "SrcMask", args, 7, true);
    auto OutLinearW = GetVarBaseListFromArgs(op_type, "OutLinearW", args, 8, false);
    auto OutLinearBias = GetVarBaseListFromArgs(op_type, "OutLinearBias", args, 9, true);
    auto FFNLnScale = GetVarBaseListFromArgs(op_type, "FFNLnScale", args, 10, false);
    auto FFNLnBias = GetVarBaseListFromArgs(op_type, "FFNLnBias", args, 11, false);
    auto FFN1Weight = GetVarBaseListFromArgs(op_type, "FFN1Weight", args, 12, false);
    auto FFN1Bias = GetVarBaseListFromArgs(op_type, "FFN1Bias", args, 13, true);
    auto FFN2Weight = GetVarBaseListFromArgs(op_type, "FFN2Weight", args, 14, false);
    auto FFN2Bias = GetVarBaseListFromArgs(op_type, "FFN2Bias", args, 15, true);
    auto QKVOutScale = GetVarBaseFromArgs(op_type, "QKVOutScale", args, 16, true);
    auto OutLinearOutScale = GetVarBaseFromArgs(op_type, "OutLinearOutScale", args, 17, true);
    auto FFN1OutScale = GetVarBaseFromArgs(op_type, "FFN1OutScale", args, 18, true);
    auto FFN2OutScale = GetVarBaseFromArgs(op_type, "FFN2OutScale", args, 19, true);
    auto CacheKVOut = GetVarBaseListFromArgs(op_type, "CacheKVOut", args, 20, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 21, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"LnScale", LnScale},{"LnBias", LnBias},{"QKVW", QKVW},{"OutLinearW", OutLinearW},{"FFNLnScale", FFNLnScale},{"FFNLnBias", FFNLnBias},{"FFN1Weight", FFN1Weight},{"FFN2Weight", FFN2Weight}};
    
    if (QKVBias.size() != 0) {
      ins["QKVBias"] = QKVBias;
    }

    if (CacheKV.size() != 0) {
      ins["CacheKV"] = CacheKV;
    }

    if (TimeStep != nullptr) {
      ins["TimeStep"] = {TimeStep};
    }

    if (SrcMask != nullptr) {
      ins["SrcMask"] = {SrcMask};
    }

    if (OutLinearBias.size() != 0) {
      ins["OutLinearBias"] = OutLinearBias;
    }

    if (FFN1Bias.size() != 0) {
      ins["FFN1Bias"] = FFN1Bias;
    }

    if (FFN2Bias.size() != 0) {
      ins["FFN2Bias"] = FFN2Bias;
    }

    if (QKVOutScale != nullptr) {
      ins["QKVOutScale"] = {QKVOutScale};
    }

    if (OutLinearOutScale != nullptr) {
      ins["OutLinearOutScale"] = {OutLinearOutScale};
    }

    if (FFN1OutScale != nullptr) {
      ins["FFN1OutScale"] = {FFN1OutScale};
    }

    if (FFN2OutScale != nullptr) {
      ins["FFN2OutScale"] = {FFN2OutScale};
    }

    outs["CacheKVOut"] = CacheKVOut;

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["CacheKVOut"],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_mine_hard_examples(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "mine_hard_examples";
    platform::RecordEvent op_type_record_event("mine_hard_examples pybind_imperative_func");
    
    auto ClsLoss = GetVarBaseFromArgs(op_type, "ClsLoss", args, 0, false);
    auto MatchIndices = GetVarBaseFromArgs(op_type, "MatchIndices", args, 1, false);
    auto MatchDist = GetVarBaseFromArgs(op_type, "MatchDist", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"NegIndices", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"UpdatedMatchIndices", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"ClsLoss", {ClsLoss}},{"MatchIndices", {MatchIndices}},{"MatchDist", {MatchDist}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["NegIndices"][0],outs["UpdatedMatchIndices"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_target_assign(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "target_assign";
    platform::RecordEvent op_type_record_event("target_assign pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto MatchIndices = GetVarBaseFromArgs(op_type, "MatchIndices", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutWeight", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"MatchIndices", {MatchIndices}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["OutWeight"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_lstm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "lstm";
    platform::RecordEvent op_type_record_event("lstm pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Weight = GetVarBaseFromArgs(op_type, "Weight", args, 1, false);
    auto Bias = GetVarBaseFromArgs(op_type, "Bias", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Hidden", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Cell", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchGate", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"BatchCellPreAct", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Weight", {Weight}},{"Bias", {Bias}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Hidden"][0],outs["Cell"][0],outs["BatchGate"][0],outs["BatchCellPreAct"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_assign_pos(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "assign_pos";
    platform::RecordEvent op_type_record_event("assign_pos pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto cum_count = GetVarBaseFromArgs(op_type, "cum_count", args, 1, false);
    auto eff_num_len = GetVarBaseFromArgs(op_type, "eff_num_len", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"cum_count", {cum_count}},{"eff_num_len", {eff_num_len}}};
    
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

static PyObject * imperative_truncated_gaussian_random(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "truncated_gaussian_random";
    platform::RecordEvent op_type_record_event("truncated_gaussian_random pybind_imperative_func");
    
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

static PyObject * imperative_match_matrix_tensor(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "match_matrix_tensor";
    platform::RecordEvent op_type_record_event("match_matrix_tensor pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Y = GetVarBaseFromArgs(op_type, "Y", args, 1, false);
    auto W = GetVarBaseFromArgs(op_type, "W", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Tmp", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Y", {Y}},{"W", {W}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Tmp"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_elementwise_div(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_div";
    platform::RecordEvent op_type_record_event("elementwise_div pybind_imperative_func");
    
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

static PyObject * imperative_kldiv_loss(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "kldiv_loss";
    platform::RecordEvent op_type_record_event("kldiv_loss pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Target = GetVarBaseFromArgs(op_type, "Target", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Loss", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Target", {Target}}};
    
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

static PyObject * imperative_cumsum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cumsum";
    platform::RecordEvent op_type_record_event("cumsum pybind_imperative_func");
    
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

static PyObject * imperative_sum(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sum";
    platform::RecordEvent op_type_record_event("sum pybind_imperative_func");
    
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

static PyObject * imperative_sum_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sum";
    platform::RecordEvent op_type_record_event("sum pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    PADDLE_ENFORCE_EQ(
      X[0]->IsLeaf() && !X[0]->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X[0]->Name()));
    X[0]->BumpInplaceVersion();
    VLOG(3) << "Var(" << X[0]->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Out", {X[0]}}};
    imperative::NameVarBaseMap ins = {{"X", X}};
    
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

static PyObject * imperative_triu_indices(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "triu_indices";
    platform::RecordEvent op_type_record_event("triu_indices pybind_imperative_func");
    
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 0, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {};
    
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

static PyMethodDef ExtestMethods[] = {
  {"fused_embedding_seq_pool", (PyCFunction)(void(*)(void))imperative_fused_embedding_seq_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_embedding_seq_pool in dygraph."},
  {"kthvalue", (PyCFunction)(void(*)(void))imperative_kthvalue, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kthvalue in dygraph."},
  {"graph_send_uv", (PyCFunction)(void(*)(void))imperative_graph_send_uv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for graph_send_uv in dygraph."},
  {"erf", (PyCFunction)(void(*)(void))imperative_erf, METH_VARARGS | METH_KEYWORDS, "C++ interface function for erf in dygraph."},
  {"yolo_box_post", (PyCFunction)(void(*)(void))imperative_yolo_box_post, METH_VARARGS | METH_KEYWORDS, "C++ interface function for yolo_box_post in dygraph."},
  {"conv2d_inception_fusion", (PyCFunction)(void(*)(void))imperative_conv2d_inception_fusion, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d_inception_fusion in dygraph."},
  {"logsumexp", (PyCFunction)(void(*)(void))imperative_logsumexp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logsumexp in dygraph."},
  {"trilinear_interp", (PyCFunction)(void(*)(void))imperative_trilinear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for trilinear_interp in dygraph."},
  {"fusion_seqpool_concat", (PyCFunction)(void(*)(void))imperative_fusion_seqpool_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_seqpool_concat in dygraph."},
  {"alloc_float_status", (PyCFunction)(void(*)(void))imperative_alloc_float_status, METH_VARARGS | METH_KEYWORDS, "C++ interface function for alloc_float_status in dygraph."},
  {"sequence_concat", (PyCFunction)(void(*)(void))imperative_sequence_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_concat in dygraph."},
  {"fusion_seqpool_cvm_concat", (PyCFunction)(void(*)(void))imperative_fusion_seqpool_cvm_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_seqpool_cvm_concat in dygraph."},
  {"unpool3d", (PyCFunction)(void(*)(void))imperative_unpool3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unpool3d in dygraph."},
  {"similarity_focus", (PyCFunction)(void(*)(void))imperative_similarity_focus, METH_VARARGS | METH_KEYWORDS, "C++ interface function for similarity_focus in dygraph."},
  {"argsort", (PyCFunction)(void(*)(void))imperative_argsort, METH_VARARGS | METH_KEYWORDS, "C++ interface function for argsort in dygraph."},
  {"sequence_expand", (PyCFunction)(void(*)(void))imperative_sequence_expand, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_expand in dygraph."},
  {"fused_bn_add_activation", (PyCFunction)(void(*)(void))imperative_fused_bn_add_activation, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_bn_add_activation in dygraph."},
  {"sgd", (PyCFunction)(void(*)(void))imperative_sgd, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sgd in dygraph."},
  {"exponential", (PyCFunction)(void(*)(void))imperative_exponential, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exponential in dygraph."},
  {"exponential_", (PyCFunction)(void(*)(void))imperative_exponential_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for exponential_ in dygraph."},
  {"bilinear_interp_v2", (PyCFunction)(void(*)(void))imperative_bilinear_interp_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilinear_interp_v2 in dygraph."},
  {"atanh", (PyCFunction)(void(*)(void))imperative_atanh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atanh in dygraph."},
  {"clip", (PyCFunction)(void(*)(void))imperative_clip, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip in dygraph."},
  {"clip_", (PyCFunction)(void(*)(void))imperative_clip_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip_ in dygraph."},
  {"sparse_to_dense", (PyCFunction)(void(*)(void))imperative_sparse_to_dense, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_to_dense in dygraph."},
  {"deformable_conv_v1", (PyCFunction)(void(*)(void))imperative_deformable_conv_v1, METH_VARARGS | METH_KEYWORDS, "C++ interface function for deformable_conv_v1 in dygraph."},
  {"hinge_loss", (PyCFunction)(void(*)(void))imperative_hinge_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for hinge_loss in dygraph."},
  {"determinant", (PyCFunction)(void(*)(void))imperative_determinant, METH_VARARGS | METH_KEYWORDS, "C++ interface function for determinant in dygraph."},
  {"conv2d_transpose", (PyCFunction)(void(*)(void))imperative_conv2d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conv2d_transpose in dygraph."},
  {"memcpy_d2h", (PyCFunction)(void(*)(void))imperative_memcpy_d2h, METH_VARARGS | METH_KEYWORDS, "C++ interface function for memcpy_d2h in dygraph."},
  {"softsign", (PyCFunction)(void(*)(void))imperative_softsign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softsign in dygraph."},
  {"fake_quantize_dequantize_abs_max", (PyCFunction)(void(*)(void))imperative_fake_quantize_dequantize_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_dequantize_abs_max in dygraph."},
  {"broadcast_tensors", (PyCFunction)(void(*)(void))imperative_broadcast_tensors, METH_VARARGS | METH_KEYWORDS, "C++ interface function for broadcast_tensors in dygraph."},
  {"cholesky_solve", (PyCFunction)(void(*)(void))imperative_cholesky_solve, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cholesky_solve in dygraph."},
  {"grid_sampler", (PyCFunction)(void(*)(void))imperative_grid_sampler, METH_VARARGS | METH_KEYWORDS, "C++ interface function for grid_sampler in dygraph."},
  {"pyramid_hash", (PyCFunction)(void(*)(void))imperative_pyramid_hash, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pyramid_hash in dygraph."},
  {"fft_c2r", (PyCFunction)(void(*)(void))imperative_fft_c2r, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_c2r in dygraph."},
  {"fake_quantize_dequantize_moving_average_abs_max", (PyCFunction)(void(*)(void))imperative_fake_quantize_dequantize_moving_average_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_dequantize_moving_average_abs_max in dygraph."},
  {"multi_dot", (PyCFunction)(void(*)(void))imperative_multi_dot, METH_VARARGS | METH_KEYWORDS, "C++ interface function for multi_dot in dygraph."},
  {"sequence_pool", (PyCFunction)(void(*)(void))imperative_sequence_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_pool in dygraph."},
  {"transpose", (PyCFunction)(void(*)(void))imperative_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for transpose in dygraph."},
  {"top_k", (PyCFunction)(void(*)(void))imperative_top_k, METH_VARARGS | METH_KEYWORDS, "C++ interface function for top_k in dygraph."},
  {"renorm", (PyCFunction)(void(*)(void))imperative_renorm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for renorm in dygraph."},
  {"pixel_unshuffle", (PyCFunction)(void(*)(void))imperative_pixel_unshuffle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pixel_unshuffle in dygraph."},
  {"take_along_axis", (PyCFunction)(void(*)(void))imperative_take_along_axis, METH_VARARGS | METH_KEYWORDS, "C++ interface function for take_along_axis in dygraph."},
  {"dist", (PyCFunction)(void(*)(void))imperative_dist, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dist in dygraph."},
  {"affine_grid", (PyCFunction)(void(*)(void))imperative_affine_grid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for affine_grid in dygraph."},
  {"gaussian_random_batch_size_like", (PyCFunction)(void(*)(void))imperative_gaussian_random_batch_size_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gaussian_random_batch_size_like in dygraph."},
  {"fake_channel_wise_dequantize_max_abs", (PyCFunction)(void(*)(void))imperative_fake_channel_wise_dequantize_max_abs, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_channel_wise_dequantize_max_abs in dygraph."},
  {"reciprocal", (PyCFunction)(void(*)(void))imperative_reciprocal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reciprocal in dygraph."},
  {"reciprocal_", (PyCFunction)(void(*)(void))imperative_reciprocal_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for reciprocal_ in dygraph."},
  {"sequence_mask", (PyCFunction)(void(*)(void))imperative_sequence_mask, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_mask in dygraph."},
  {"prune_gate_by_capacity", (PyCFunction)(void(*)(void))imperative_prune_gate_by_capacity, METH_VARARGS | METH_KEYWORDS, "C++ interface function for prune_gate_by_capacity in dygraph."},
  {"fill_diagonal_tensor", (PyCFunction)(void(*)(void))imperative_fill_diagonal_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_tensor in dygraph."},
  {"fill_diagonal_tensor_", (PyCFunction)(void(*)(void))imperative_fill_diagonal_tensor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_diagonal_tensor_ in dygraph."},
  {"abs", (PyCFunction)(void(*)(void))imperative_abs, METH_VARARGS | METH_KEYWORDS, "C++ interface function for abs in dygraph."},
  {"partial_concat", (PyCFunction)(void(*)(void))imperative_partial_concat, METH_VARARGS | METH_KEYWORDS, "C++ interface function for partial_concat in dygraph."},
  {"elu", (PyCFunction)(void(*)(void))imperative_elu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elu in dygraph."},
  {"elu_", (PyCFunction)(void(*)(void))imperative_elu_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elu_ in dygraph."},
  {"index_select", (PyCFunction)(void(*)(void))imperative_index_select, METH_VARARGS | METH_KEYWORDS, "C++ interface function for index_select in dygraph."},
  {"row_conv", (PyCFunction)(void(*)(void))imperative_row_conv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for row_conv in dygraph."},
  {"cross", (PyCFunction)(void(*)(void))imperative_cross, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cross in dygraph."},
  {"elementwise_mul", (PyCFunction)(void(*)(void))imperative_elementwise_mul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_mul in dygraph."},
  {"decayed_adagrad", (PyCFunction)(void(*)(void))imperative_decayed_adagrad, METH_VARARGS | METH_KEYWORDS, "C++ interface function for decayed_adagrad in dygraph."},
  {"bipartite_match", (PyCFunction)(void(*)(void))imperative_bipartite_match, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bipartite_match in dygraph."},
  {"run_program", (PyCFunction)(void(*)(void))imperative_run_program, METH_VARARGS | METH_KEYWORDS, "C++ interface function for run_program in dygraph."},
  {"fake_quantize_moving_average_abs_max", (PyCFunction)(void(*)(void))imperative_fake_quantize_moving_average_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_moving_average_abs_max in dygraph."},
  {"fused_multi_transformer_int8", (PyCFunction)(void(*)(void))imperative_fused_multi_transformer_int8, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_multi_transformer_int8 in dygraph."},
  {"mine_hard_examples", (PyCFunction)(void(*)(void))imperative_mine_hard_examples, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mine_hard_examples in dygraph."},
  {"target_assign", (PyCFunction)(void(*)(void))imperative_target_assign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for target_assign in dygraph."},
  {"lstm", (PyCFunction)(void(*)(void))imperative_lstm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for lstm in dygraph."},
  {"assign_pos", (PyCFunction)(void(*)(void))imperative_assign_pos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for assign_pos in dygraph."},
  {"truncated_gaussian_random", (PyCFunction)(void(*)(void))imperative_truncated_gaussian_random, METH_VARARGS | METH_KEYWORDS, "C++ interface function for truncated_gaussian_random in dygraph."},
  {"match_matrix_tensor", (PyCFunction)(void(*)(void))imperative_match_matrix_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for match_matrix_tensor in dygraph."},
  {"elementwise_div", (PyCFunction)(void(*)(void))imperative_elementwise_div, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_div in dygraph."},
  {"kldiv_loss", (PyCFunction)(void(*)(void))imperative_kldiv_loss, METH_VARARGS | METH_KEYWORDS, "C++ interface function for kldiv_loss in dygraph."},
  {"cumsum", (PyCFunction)(void(*)(void))imperative_cumsum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumsum in dygraph."},
  {"sum", (PyCFunction)(void(*)(void))imperative_sum, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sum in dygraph."},
  {"sum_", (PyCFunction)(void(*)(void))imperative_sum_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sum_ in dygraph."},
  {"triu_indices", (PyCFunction)(void(*)(void))imperative_triu_indices, METH_VARARGS | METH_KEYWORDS, "C++ interface function for triu_indices in dygraph."},
  {nullptr,nullptr,0,nullptr}};

void BindOpFunctions7(pybind11::module *module) {
  auto m = module->def_submodule("ops");
  if (PyModule_AddFunctions(m.ptr(), ExtestMethods) < 0) {
    PADDLE_THROW(platform::errors::Fatal ("Add functions to core.ops failed!"));
  }

  InitOpsAttrTypeMap();}

} // namespace pybind
} // namespace paddle
