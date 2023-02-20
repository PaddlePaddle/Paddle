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

static PyObject * imperative_acos(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "acos";
    platform::RecordEvent op_type_record_event("acos pybind_imperative_func");
    
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

static PyObject * imperative_unpool(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "unpool";
    platform::RecordEvent op_type_record_event("unpool pybind_imperative_func");
    
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

static PyObject * imperative_cumprod(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "cumprod";
    platform::RecordEvent op_type_record_event("cumprod pybind_imperative_func");
    
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

static PyObject * imperative_sample_logits(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sample_logits";
    platform::RecordEvent op_type_record_event("sample_logits pybind_imperative_func");
    
    auto Logits = GetVarBaseFromArgs(op_type, "Logits", args, 0, false);
    auto Labels = GetVarBaseFromArgs(op_type, "Labels", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Samples", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Probabilities", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LogitsDim", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"LabelsDim", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SampledLogits", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SampledLabels", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Logits", {Logits}},{"Labels", {Labels}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Samples"][0],outs["Probabilities"][0],outs["LogitsDim"][0],outs["LabelsDim"][0],outs["SampledLogits"][0],outs["SampledLabels"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_pull_box_extended_sparse(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pull_box_extended_sparse";
    platform::RecordEvent op_type_record_event("pull_box_extended_sparse pybind_imperative_func");
    
    auto Ids = GetVarBaseListFromArgs(op_type, "Ids", args, 0, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 1, false);
    auto OutExtendNum = GetUnsignedLongFromArgs(op_type, "OutExtendNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)},{"OutExtend", ConstructDuplicableOutput(OutExtendNum)}};
    imperative::NameVarBaseMap ins = {{"Ids", Ids}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"],outs["OutExtend"]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sparse_sparse_coo_tensor(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sparse_sparse_coo_tensor";
    platform::RecordEvent op_type_record_event("sparse_sparse_coo_tensor pybind_imperative_func");
    
    auto values = GetVarBaseFromArgs(op_type, "values", args, 0, false);
    auto indices = GetVarBaseFromArgs(op_type, "indices", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"values", {values}},{"indices", {indices}}};
    
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

static PyObject * imperative_crop_tensor(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "crop_tensor";
    platform::RecordEvent op_type_record_event("crop_tensor pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Shape = GetVarBaseFromArgs(op_type, "Shape", args, 1, true);
    auto Offsets = GetVarBaseFromArgs(op_type, "Offsets", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    if (Shape != nullptr) {
      ins["Shape"] = {Shape};
    }

    if (Offsets != nullptr) {
      ins["Offsets"] = {Offsets};
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

static PyObject * imperative_fill_constant(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_constant";
    platform::RecordEvent op_type_record_event("fill_constant pybind_imperative_func");
    
    auto Out = GetVarBaseFromArgs(op_type, "Out", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {Out}}};
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

static PyObject * imperative_deformable_conv(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "deformable_conv";
    platform::RecordEvent op_type_record_event("deformable_conv pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Offset = GetVarBaseFromArgs(op_type, "Offset", args, 1, false);
    auto Mask = GetVarBaseFromArgs(op_type, "Mask", args, 2, false);
    auto Filter = GetVarBaseFromArgs(op_type, "Filter", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Output", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Offset", {Offset}},{"Mask", {Mask}},{"Filter", {Filter}}};
    
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

static PyObject * imperative_generate_mask_labels(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "generate_mask_labels";
    platform::RecordEvent op_type_record_event("generate_mask_labels pybind_imperative_func");
    
    auto ImInfo = GetVarBaseFromArgs(op_type, "ImInfo", args, 0, false);
    auto GtClasses = GetVarBaseFromArgs(op_type, "GtClasses", args, 1, false);
    auto IsCrowd = GetVarBaseFromArgs(op_type, "IsCrowd", args, 2, false);
    auto GtSegms = GetVarBaseFromArgs(op_type, "GtSegms", args, 3, false);
    auto Rois = GetVarBaseFromArgs(op_type, "Rois", args, 4, false);
    auto LabelsInt32 = GetVarBaseFromArgs(op_type, "LabelsInt32", args, 5, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 6, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"MaskRois", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RoiHasMaskInt32", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"MaskInt32", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"ImInfo", {ImInfo}},{"GtClasses", {GtClasses}},{"IsCrowd", {IsCrowd}},{"GtSegms", {GtSegms}},{"Rois", {Rois}},{"LabelsInt32", {LabelsInt32}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["MaskRois"][0],outs["RoiHasMaskInt32"][0],outs["MaskInt32"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_locality_aware_nms(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "locality_aware_nms";
    platform::RecordEvent op_type_record_event("locality_aware_nms pybind_imperative_func");
    
    auto BBoxes = GetVarBaseFromArgs(op_type, "BBoxes", args, 0, false);
    auto Scores = GetVarBaseFromArgs(op_type, "Scores", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"BBoxes", {BBoxes}},{"Scores", {Scores}}};
    
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

static PyObject * imperative_expand_as(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "expand_as";
    platform::RecordEvent op_type_record_event("expand_as pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto target_tensor = GetVarBaseFromArgs(op_type, "target_tensor", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"target_tensor", {target_tensor}}};
    
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

static PyObject * imperative_matrix_power(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "matrix_power";
    platform::RecordEvent op_type_record_event("matrix_power pybind_imperative_func");
    
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

static PyObject * imperative_greater_equal(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "greater_equal";
    platform::RecordEvent op_type_record_event("greater_equal pybind_imperative_func");
    
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

static PyObject * imperative_generate_proposals(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "generate_proposals";
    platform::RecordEvent op_type_record_event("generate_proposals pybind_imperative_func");
    
    auto Scores = GetVarBaseFromArgs(op_type, "Scores", args, 0, false);
    auto BboxDeltas = GetVarBaseFromArgs(op_type, "BboxDeltas", args, 1, false);
    auto ImInfo = GetVarBaseFromArgs(op_type, "ImInfo", args, 2, false);
    auto Anchors = GetVarBaseFromArgs(op_type, "Anchors", args, 3, false);
    auto Variances = GetVarBaseFromArgs(op_type, "Variances", args, 4, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 5, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"RpnRois", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RpnRoiProbs", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"RpnRoisNum", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Scores", {Scores}},{"BboxDeltas", {BboxDeltas}},{"ImInfo", {ImInfo}},{"Anchors", {Anchors}},{"Variances", {Variances}}};
    
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

static PyObject * imperative_number_count(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "number_count";
    platform::RecordEvent op_type_record_event("number_count pybind_imperative_func");
    
    auto numbers = GetVarBaseFromArgs(op_type, "numbers", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"numbers", {numbers}}};
    
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

static PyObject * imperative_bilinear_interp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "bilinear_interp";
    platform::RecordEvent op_type_record_event("bilinear_interp pybind_imperative_func");
    
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

static PyObject * imperative_distributed_fused_lamb(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "distributed_fused_lamb";
    platform::RecordEvent op_type_record_event("distributed_fused_lamb pybind_imperative_func");
    
    auto Param = GetVarBaseListFromArgs(op_type, "Param", args, 0, false);
    auto Grad = GetVarBaseListFromArgs(op_type, "Grad", args, 1, false);
    auto Moment1 = GetVarBaseFromArgs(op_type, "Moment1", args, 2, false);
    auto Moment2 = GetVarBaseFromArgs(op_type, "Moment2", args, 3, false);
    auto Beta1Pow = GetVarBaseFromArgs(op_type, "Beta1Pow", args, 4, false);
    auto Beta2Pow = GetVarBaseFromArgs(op_type, "Beta2Pow", args, 5, false);
    auto FusedParamOffsets = GetVarBaseFromArgs(op_type, "FusedParamOffsets", args, 6, false);
    auto FP32ShardFusedParamOffsets = GetVarBaseFromArgs(op_type, "FP32ShardFusedParamOffsets", args, 7, false);
    auto FP16ShardFusedParamOffsets = GetVarBaseFromArgs(op_type, "FP16ShardFusedParamOffsets", args, 8, false);
    auto ParamInfo = GetVarBaseFromArgs(op_type, "ParamInfo", args, 9, false);
    auto ParamOrder = GetVarBaseFromArgs(op_type, "ParamOrder", args, 10, false);
    auto LearningRate = GetVarBaseFromArgs(op_type, "LearningRate", args, 11, false);
    auto GlobalScale = GetVarBaseFromArgs(op_type, "GlobalScale", args, 12, false);
    auto ParamOutNum = GetUnsignedLongFromArgs(op_type, "ParamOutNum", args, 13, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 14, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Moment1Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Moment2Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Beta1PowOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Beta2PowOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ParamOut", ConstructDuplicableOutput(ParamOutNum)},{"FoundInf", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Step", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Param", Param},{"Grad", Grad},{"Moment1", {Moment1}},{"Moment2", {Moment2}},{"Beta1Pow", {Beta1Pow}},{"Beta2Pow", {Beta2Pow}},{"FusedParamOffsets", {FusedParamOffsets}},{"FP32ShardFusedParamOffsets", {FP32ShardFusedParamOffsets}},{"FP16ShardFusedParamOffsets", {FP16ShardFusedParamOffsets}},{"ParamInfo", {ParamInfo}},{"ParamOrder", {ParamOrder}},{"LearningRate", {LearningRate}},{"GlobalScale", {GlobalScale}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Moment1Out"][0],outs["Moment2Out"][0],outs["Beta1PowOut"][0],outs["Beta2PowOut"][0],outs["ParamOut"],outs["FoundInf"][0],outs["Step"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_sigmoid(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sigmoid";
    platform::RecordEvent op_type_record_event("sigmoid pybind_imperative_func");
    
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

static PyObject * imperative_sigmoid_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sigmoid";
    platform::RecordEvent op_type_record_event("sigmoid pybind_imperative_func");
    
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

static PyObject * imperative_inplace_abn(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "inplace_abn";
    platform::RecordEvent op_type_record_event("inplace_abn pybind_imperative_func");
    
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

static PyObject * imperative_inplace_abn_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "inplace_abn";
    platform::RecordEvent op_type_record_event("inplace_abn pybind_imperative_func");
    
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
    
    PADDLE_ENFORCE_EQ(
      X->IsLeaf() && !X->OverridedStopGradient(), false,
      platform::errors::InvalidArgument("Leaf Var (%s) that doesn't stop gradient can't use inplace strategy.", X->Name()));
    X->BumpInplaceVersion();
    VLOG(3) << "Var(" << X->Name() << ") uses Inplace Strategy.";

    imperative::NameVarBaseMap outs = {{"Y", {X}},{"MeanOut", {MeanOut}},{"VarianceOut", {VarianceOut}},{"SavedMean", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"SavedVariance", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"ReserveSpace", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Scale", {Scale}},{"Bias", {Bias}},{"Mean", {Mean}},{"Variance", {Variance}}};
    
    if (MomentumTensor != nullptr) {
      ins["MomentumTensor"] = {MomentumTensor};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {{"X", "Y"}});
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

static PyObject * imperative_softshrink(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "softshrink";
    platform::RecordEvent op_type_record_event("softshrink pybind_imperative_func");
    
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

static PyObject * imperative_mul(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "mul";
    platform::RecordEvent op_type_record_event("mul pybind_imperative_func");
    
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

static PyObject * imperative_data_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "data_norm";
    platform::RecordEvent op_type_record_event("data_norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto BatchSize = GetVarBaseFromArgs(op_type, "BatchSize", args, 1, false);
    auto BatchSum = GetVarBaseFromArgs(op_type, "BatchSum", args, 2, false);
    auto BatchSquareSum = GetVarBaseFromArgs(op_type, "BatchSquareSum", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Y", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Means", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Scales", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"BatchSize", {BatchSize}},{"BatchSum", {BatchSum}},{"BatchSquareSum", {BatchSquareSum}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Y"][0],outs["Means"][0],outs["Scales"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fused_multi_transformer(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_multi_transformer";
    platform::RecordEvent op_type_record_event("fused_multi_transformer pybind_imperative_func");
    
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
    auto CacheKVOut = GetVarBaseListFromArgs(op_type, "CacheKVOut", args, 16, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 17, PyTuple_GET_SIZE(args) , attrs);
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

static PyObject * imperative_asinh(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "asinh";
    platform::RecordEvent op_type_record_event("asinh pybind_imperative_func");
    
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

static PyObject * imperative_get_tensor_from_selected_rows(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "get_tensor_from_selected_rows";
    platform::RecordEvent op_type_record_event("get_tensor_from_selected_rows pybind_imperative_func");
    
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

static PyObject * imperative_spp(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "spp";
    platform::RecordEvent op_type_record_event("spp pybind_imperative_func");
    
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

static PyObject * imperative_floor(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "floor";
    platform::RecordEvent op_type_record_event("floor pybind_imperative_func");
    
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

static PyObject * imperative_floor_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "floor";
    platform::RecordEvent op_type_record_event("floor pybind_imperative_func");
    
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

static PyObject * imperative_as_real(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "as_real";
    platform::RecordEvent op_type_record_event("as_real pybind_imperative_func");
    
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

static PyObject * imperative_gelu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "gelu";
    platform::RecordEvent op_type_record_event("gelu pybind_imperative_func");
    
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

static PyObject * imperative_retinanet_detection_output(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "retinanet_detection_output";
    platform::RecordEvent op_type_record_event("retinanet_detection_output pybind_imperative_func");
    
    auto BBoxes = GetVarBaseListFromArgs(op_type, "BBoxes", args, 0, false);
    auto Scores = GetVarBaseListFromArgs(op_type, "Scores", args, 1, false);
    auto Anchors = GetVarBaseListFromArgs(op_type, "Anchors", args, 2, false);
    auto ImInfo = GetVarBaseFromArgs(op_type, "ImInfo", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"BBoxes", BBoxes},{"Scores", Scores},{"Anchors", Anchors},{"ImInfo", {ImInfo}}};
    
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

static PyObject * imperative_minus(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "minus";
    platform::RecordEvent op_type_record_event("minus pybind_imperative_func");
    
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

static PyObject * imperative_push_dense(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "push_dense";
    platform::RecordEvent op_type_record_event("push_dense pybind_imperative_func");
    
    auto Ids = GetVarBaseListFromArgs(op_type, "Ids", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {};
    imperative::NameVarBaseMap ins = {{"Ids", Ids}};
    
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

static PyObject * imperative_silu(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "silu";
    platform::RecordEvent op_type_record_event("silu pybind_imperative_func");
    
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

static PyObject * imperative_sequence_erase(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sequence_erase";
    platform::RecordEvent op_type_record_event("sequence_erase pybind_imperative_func");
    
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

static PyObject * imperative_real(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "real";
    platform::RecordEvent op_type_record_event("real pybind_imperative_func");
    
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

static PyObject * imperative_nearest_interp_v2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "nearest_interp_v2";
    platform::RecordEvent op_type_record_event("nearest_interp_v2 pybind_imperative_func");
    
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

static PyObject * imperative_dgc_clip_by_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dgc_clip_by_norm";
    platform::RecordEvent op_type_record_event("dgc_clip_by_norm pybind_imperative_func");
    
    auto current_step = GetVarBaseFromArgs(op_type, "current_step", args, 0, false);
    auto X = GetVarBaseFromArgs(op_type, "X", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"current_step", {current_step}},{"X", {X}}};
    
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

static PyObject * imperative_squeeze2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "squeeze2";
    platform::RecordEvent op_type_record_event("squeeze2 pybind_imperative_func");
    
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

static PyObject * imperative_squeeze2_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "squeeze2";
    platform::RecordEvent op_type_record_event("squeeze2 pybind_imperative_func");
    
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

static PyObject * imperative_conj(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "conj";
    platform::RecordEvent op_type_record_event("conj pybind_imperative_func");
    
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

static PyObject * imperative_strided_slice(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "strided_slice";
    platform::RecordEvent op_type_record_event("strided_slice pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto StartsTensor = GetVarBaseFromArgs(op_type, "StartsTensor", args, 1, true);
    auto EndsTensor = GetVarBaseFromArgs(op_type, "EndsTensor", args, 2, true);
    auto StridesTensor = GetVarBaseFromArgs(op_type, "StridesTensor", args, 3, true);
    auto StartsTensorList = GetVarBaseListFromArgs(op_type, "StartsTensorList", args, 4, true);
    auto EndsTensorList = GetVarBaseListFromArgs(op_type, "EndsTensorList", args, 5, true);
    auto StridesTensorList = GetVarBaseListFromArgs(op_type, "StridesTensorList", args, 6, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 7, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}}};
    
    if (StartsTensor != nullptr) {
      ins["StartsTensor"] = {StartsTensor};
    }

    if (EndsTensor != nullptr) {
      ins["EndsTensor"] = {EndsTensor};
    }

    if (StridesTensor != nullptr) {
      ins["StridesTensor"] = {StridesTensor};
    }

    if (StartsTensorList.size() != 0) {
      ins["StartsTensorList"] = StartsTensorList;
    }

    if (EndsTensorList.size() != 0) {
      ins["EndsTensorList"] = EndsTensorList;
    }

    if (StridesTensorList.size() != 0) {
      ins["StridesTensorList"] = StridesTensorList;
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

static PyObject * imperative_precision_recall(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "precision_recall";
    platform::RecordEvent op_type_record_event("precision_recall pybind_imperative_func");
    
    auto MaxProbs = GetVarBaseFromArgs(op_type, "MaxProbs", args, 0, false);
    auto Indices = GetVarBaseFromArgs(op_type, "Indices", args, 1, false);
    auto Labels = GetVarBaseFromArgs(op_type, "Labels", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"BatchMetrics", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AccumMetrics", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"AccumStatesInfo", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"MaxProbs", {MaxProbs}},{"Indices", {Indices}},{"Labels", {Labels}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["BatchMetrics"][0],outs["AccumMetrics"][0],outs["AccumStatesInfo"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fusion_seqexpand_concat_fc(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fusion_seqexpand_concat_fc";
    platform::RecordEvent op_type_record_event("fusion_seqexpand_concat_fc pybind_imperative_func");
    
    auto X = GetVarBaseListFromArgs(op_type, "X", args, 0, false);
    auto FCWeight = GetVarBaseFromArgs(op_type, "FCWeight", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"FCOut", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", X},{"FCWeight", {FCWeight}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["FCOut"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_save(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "save";
    platform::RecordEvent op_type_record_event("save pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
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

static PyObject * imperative_depthwise_conv2d_transpose(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "depthwise_conv2d_transpose";
    platform::RecordEvent op_type_record_event("depthwise_conv2d_transpose pybind_imperative_func");
    
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

static PyObject * imperative_fake_quantize_range_abs_max(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fake_quantize_range_abs_max";
    platform::RecordEvent op_type_record_event("fake_quantize_range_abs_max pybind_imperative_func");
    
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

static PyObject * imperative_positive_negative_pair(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "positive_negative_pair";
    platform::RecordEvent op_type_record_event("positive_negative_pair pybind_imperative_func");
    
    auto Score = GetVarBaseFromArgs(op_type, "Score", args, 0, false);
    auto Label = GetVarBaseFromArgs(op_type, "Label", args, 1, false);
    auto QueryID = GetVarBaseFromArgs(op_type, "QueryID", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"PositivePair", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"NegativePair", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Score", {Score}},{"Label", {Label}},{"QueryID", {QueryID}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["PositivePair"][0],outs["NegativePair"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_square(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "square";
    platform::RecordEvent op_type_record_event("square pybind_imperative_func");
    
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

static PyObject * imperative_square_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "square";
    platform::RecordEvent op_type_record_event("square pybind_imperative_func");
    
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

static PyObject * imperative_var_conv_2d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "var_conv_2d";
    platform::RecordEvent op_type_record_event("var_conv_2d pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ROW = GetVarBaseFromArgs(op_type, "ROW", args, 1, false);
    auto COLUMN = GetVarBaseFromArgs(op_type, "COLUMN", args, 2, false);
    auto W = GetVarBaseFromArgs(op_type, "W", args, 3, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 4, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Col", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"ROW", {ROW}},{"COLUMN", {COLUMN}},{"W", {W}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Col"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_log1p(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "log1p";
    platform::RecordEvent op_type_record_event("log1p pybind_imperative_func");
    
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

static PyObject * imperative_channel_shuffle(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "channel_shuffle";
    platform::RecordEvent op_type_record_event("channel_shuffle pybind_imperative_func");
    
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

static PyObject * imperative_sparse_indices(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "sparse_indices";
    platform::RecordEvent op_type_record_event("sparse_indices pybind_imperative_func");
    
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

static PyObject * imperative_atan2(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "atan2";
    platform::RecordEvent op_type_record_event("atan2 pybind_imperative_func");
    
    auto X1 = GetVarBaseFromArgs(op_type, "X1", args, 0, false);
    auto X2 = GetVarBaseFromArgs(op_type, "X2", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X1", {X1}},{"X2", {X2}}};
    
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

static PyObject * imperative_fused_softmax_mask_upper_triangle(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fused_softmax_mask_upper_triangle";
    platform::RecordEvent op_type_record_event("fused_softmax_mask_upper_triangle pybind_imperative_func");
    
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

static PyObject * imperative_clip_by_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "clip_by_norm";
    platform::RecordEvent op_type_record_event("clip_by_norm pybind_imperative_func");
    
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

static PyObject * imperative_box_decoder_and_assign(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "box_decoder_and_assign";
    platform::RecordEvent op_type_record_event("box_decoder_and_assign pybind_imperative_func");
    
    auto PriorBox = GetVarBaseFromArgs(op_type, "PriorBox", args, 0, false);
    auto TargetBox = GetVarBaseFromArgs(op_type, "TargetBox", args, 1, false);
    auto BoxScore = GetVarBaseFromArgs(op_type, "BoxScore", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"DecodeBox", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"OutputAssignBox", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"PriorBox", {PriorBox}},{"TargetBox", {TargetBox}},{"BoxScore", {BoxScore}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["DecodeBox"][0],outs["OutputAssignBox"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_roi_pool(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "roi_pool";
    platform::RecordEvent op_type_record_event("roi_pool pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto ROIs = GetVarBaseFromArgs(op_type, "ROIs", args, 1, false);
    auto RoisNum = GetVarBaseFromArgs(op_type, "RoisNum", args, 2, true);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Argmax", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"ROIs", {ROIs}}};
    
    if (RoisNum != nullptr) {
      ins["RoisNum"] = {RoisNum};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Out"][0],outs["Argmax"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_fft_r2c(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fft_r2c";
    platform::RecordEvent op_type_record_event("fft_r2c pybind_imperative_func");
    
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

static PyObject * imperative_overlap_add(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "overlap_add";
    platform::RecordEvent op_type_record_event("overlap_add pybind_imperative_func");
    
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

static PyObject * imperative_fill_constant_batch_size_like(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_constant_batch_size_like";
    platform::RecordEvent op_type_record_event("fill_constant_batch_size_like pybind_imperative_func");
    
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

static PyObject * imperative_fill_any(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_any";
    platform::RecordEvent op_type_record_event("fill_any pybind_imperative_func");
    
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

static PyObject * imperative_fill_any_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "fill_any";
    platform::RecordEvent op_type_record_event("fill_any pybind_imperative_func");
    
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

static PyObject * imperative_dequantize_log(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "dequantize_log";
    platform::RecordEvent op_type_record_event("dequantize_log pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    auto Dict = GetVarBaseFromArgs(op_type, "Dict", args, 1, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 2, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}},{"Dict", {Dict}}};
    
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

static PyObject * imperative_max_pool2d_with_index(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "max_pool2d_with_index";
    platform::RecordEvent op_type_record_event("max_pool2d_with_index pybind_imperative_func");
    
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

static PyObject * imperative_pad3d(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pad3d";
    platform::RecordEvent op_type_record_event("pad3d pybind_imperative_func");
    
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

static PyObject * imperative_norm(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "norm";
    platform::RecordEvent op_type_record_event("norm pybind_imperative_func");
    
    auto X = GetVarBaseFromArgs(op_type, "X", args, 0, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 1, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Norm", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Out", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"X", {X}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Norm"][0],outs["Out"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_viterbi_decode(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "viterbi_decode";
    platform::RecordEvent op_type_record_event("viterbi_decode pybind_imperative_func");
    
    auto Input = GetVarBaseFromArgs(op_type, "Input", args, 0, false);
    auto Transition = GetVarBaseFromArgs(op_type, "Transition", args, 1, false);
    auto Length = GetVarBaseFromArgs(op_type, "Length", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Scores", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}},{"Path", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"Input", {Input}},{"Transition", {Transition}},{"Length", {Length}}};
    
    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(std::make_tuple(outs["Scores"][0],outs["Path"][0]));
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_mish(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "mish";
    platform::RecordEvent op_type_record_event("mish pybind_imperative_func");
    
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

static PyObject * imperative_box_coder(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "box_coder";
    platform::RecordEvent op_type_record_event("box_coder pybind_imperative_func");
    
    auto PriorBox = GetVarBaseFromArgs(op_type, "PriorBox", args, 0, false);
    auto PriorBoxVar = GetVarBaseFromArgs(op_type, "PriorBoxVar", args, 1, true);
    auto TargetBox = GetVarBaseFromArgs(op_type, "TargetBox", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"OutputBox", {std::shared_ptr<imperative::VarBase>(new imperative::VarBase("auto_"+std::to_string(VarBaseUniqueNameID++)+"_"))}}};
    imperative::NameVarBaseMap ins = {{"PriorBox", {PriorBox}},{"TargetBox", {TargetBox}}};
    
    if (PriorBoxVar != nullptr) {
      ins["PriorBoxVar"] = {PriorBoxVar};
    }

    imperative::GetCurrentTracer()->TraceOp(op_type, ins, outs, attrs, {});
    PyEval_RestoreThread(tstate);
    tstate = nullptr;
    return MakeReturnPyObject(outs["OutputBox"][0]);
  }
  catch(...) {
    if (tstate) {
      PyEval_RestoreThread(tstate);
    }
    ThrowExceptionToPython(std::current_exception());
    return nullptr;
  }
}

static PyObject * imperative_flatten(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "flatten";
    platform::RecordEvent op_type_record_event("flatten pybind_imperative_func");
    
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

static PyObject * imperative_flatten_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "flatten";
    platform::RecordEvent op_type_record_event("flatten pybind_imperative_func");
    
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

static PyObject * imperative_elementwise_mod(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_mod";
    platform::RecordEvent op_type_record_event("elementwise_mod pybind_imperative_func");
    
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

static PyObject * imperative_elementwise_mod_(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "elementwise_mod";
    platform::RecordEvent op_type_record_event("elementwise_mod pybind_imperative_func");
    
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

static PyObject * imperative_margin_cross_entropy(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "margin_cross_entropy";
    platform::RecordEvent op_type_record_event("margin_cross_entropy pybind_imperative_func");
    
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

static PyObject * imperative_pull_sparse(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pull_sparse";
    platform::RecordEvent op_type_record_event("pull_sparse pybind_imperative_func");
    
    auto Ids = GetVarBaseListFromArgs(op_type, "Ids", args, 0, false);
    auto W = GetVarBaseListFromArgs(op_type, "W", args, 1, false);
    auto OutNum = GetUnsignedLongFromArgs(op_type, "OutNum", args, 2, false);
    framework::AttributeMap attrs;
    ConstructAttrMapFromPyArgs(op_type, args, 3, PyTuple_GET_SIZE(args) , attrs);
    tstate = PyEval_SaveThread();
    
    imperative::NameVarBaseMap outs = {{"Out", ConstructDuplicableOutput(OutNum)}};
    imperative::NameVarBaseMap ins = {{"Ids", Ids},{"W", W}};
    
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

static PyObject * imperative_logical_and(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "logical_and";
    platform::RecordEvent op_type_record_event("logical_and pybind_imperative_func");
    
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

static PyObject * imperative_pow(PyObject *self, PyObject *args, PyObject *kwargs)
{
  PyThreadState *tstate = nullptr;
  try
  {
    std::string op_type = "pow";
    platform::RecordEvent op_type_record_event("pow pybind_imperative_func");
    
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

static PyMethodDef ExtestMethods[] = {
  {"acos", (PyCFunction)(void(*)(void))imperative_acos, METH_VARARGS | METH_KEYWORDS, "C++ interface function for acos in dygraph."},
  {"unpool", (PyCFunction)(void(*)(void))imperative_unpool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for unpool in dygraph."},
  {"cumprod", (PyCFunction)(void(*)(void))imperative_cumprod, METH_VARARGS | METH_KEYWORDS, "C++ interface function for cumprod in dygraph."},
  {"sample_logits", (PyCFunction)(void(*)(void))imperative_sample_logits, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sample_logits in dygraph."},
  {"pull_box_extended_sparse", (PyCFunction)(void(*)(void))imperative_pull_box_extended_sparse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pull_box_extended_sparse in dygraph."},
  {"sparse_sparse_coo_tensor", (PyCFunction)(void(*)(void))imperative_sparse_sparse_coo_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_sparse_coo_tensor in dygraph."},
  {"crop_tensor", (PyCFunction)(void(*)(void))imperative_crop_tensor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for crop_tensor in dygraph."},
  {"fill_constant", (PyCFunction)(void(*)(void))imperative_fill_constant, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_constant in dygraph."},
  {"deformable_conv", (PyCFunction)(void(*)(void))imperative_deformable_conv, METH_VARARGS | METH_KEYWORDS, "C++ interface function for deformable_conv in dygraph."},
  {"generate_mask_labels", (PyCFunction)(void(*)(void))imperative_generate_mask_labels, METH_VARARGS | METH_KEYWORDS, "C++ interface function for generate_mask_labels in dygraph."},
  {"locality_aware_nms", (PyCFunction)(void(*)(void))imperative_locality_aware_nms, METH_VARARGS | METH_KEYWORDS, "C++ interface function for locality_aware_nms in dygraph."},
  {"expand_as", (PyCFunction)(void(*)(void))imperative_expand_as, METH_VARARGS | METH_KEYWORDS, "C++ interface function for expand_as in dygraph."},
  {"matrix_power", (PyCFunction)(void(*)(void))imperative_matrix_power, METH_VARARGS | METH_KEYWORDS, "C++ interface function for matrix_power in dygraph."},
  {"greater_equal", (PyCFunction)(void(*)(void))imperative_greater_equal, METH_VARARGS | METH_KEYWORDS, "C++ interface function for greater_equal in dygraph."},
  {"generate_proposals", (PyCFunction)(void(*)(void))imperative_generate_proposals, METH_VARARGS | METH_KEYWORDS, "C++ interface function for generate_proposals in dygraph."},
  {"number_count", (PyCFunction)(void(*)(void))imperative_number_count, METH_VARARGS | METH_KEYWORDS, "C++ interface function for number_count in dygraph."},
  {"bilinear_interp", (PyCFunction)(void(*)(void))imperative_bilinear_interp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for bilinear_interp in dygraph."},
  {"distributed_fused_lamb", (PyCFunction)(void(*)(void))imperative_distributed_fused_lamb, METH_VARARGS | METH_KEYWORDS, "C++ interface function for distributed_fused_lamb in dygraph."},
  {"sigmoid", (PyCFunction)(void(*)(void))imperative_sigmoid, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid in dygraph."},
  {"sigmoid_", (PyCFunction)(void(*)(void))imperative_sigmoid_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sigmoid_ in dygraph."},
  {"inplace_abn", (PyCFunction)(void(*)(void))imperative_inplace_abn, METH_VARARGS | METH_KEYWORDS, "C++ interface function for inplace_abn in dygraph."},
  {"inplace_abn_", (PyCFunction)(void(*)(void))imperative_inplace_abn_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for inplace_abn_ in dygraph."},
  {"softshrink", (PyCFunction)(void(*)(void))imperative_softshrink, METH_VARARGS | METH_KEYWORDS, "C++ interface function for softshrink in dygraph."},
  {"mul", (PyCFunction)(void(*)(void))imperative_mul, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mul in dygraph."},
  {"data_norm", (PyCFunction)(void(*)(void))imperative_data_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for data_norm in dygraph."},
  {"fused_multi_transformer", (PyCFunction)(void(*)(void))imperative_fused_multi_transformer, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_multi_transformer in dygraph."},
  {"asinh", (PyCFunction)(void(*)(void))imperative_asinh, METH_VARARGS | METH_KEYWORDS, "C++ interface function for asinh in dygraph."},
  {"get_tensor_from_selected_rows", (PyCFunction)(void(*)(void))imperative_get_tensor_from_selected_rows, METH_VARARGS | METH_KEYWORDS, "C++ interface function for get_tensor_from_selected_rows in dygraph."},
  {"spp", (PyCFunction)(void(*)(void))imperative_spp, METH_VARARGS | METH_KEYWORDS, "C++ interface function for spp in dygraph."},
  {"floor", (PyCFunction)(void(*)(void))imperative_floor, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor in dygraph."},
  {"floor_", (PyCFunction)(void(*)(void))imperative_floor_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for floor_ in dygraph."},
  {"as_real", (PyCFunction)(void(*)(void))imperative_as_real, METH_VARARGS | METH_KEYWORDS, "C++ interface function for as_real in dygraph."},
  {"gelu", (PyCFunction)(void(*)(void))imperative_gelu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for gelu in dygraph."},
  {"retinanet_detection_output", (PyCFunction)(void(*)(void))imperative_retinanet_detection_output, METH_VARARGS | METH_KEYWORDS, "C++ interface function for retinanet_detection_output in dygraph."},
  {"minus", (PyCFunction)(void(*)(void))imperative_minus, METH_VARARGS | METH_KEYWORDS, "C++ interface function for minus in dygraph."},
  {"push_dense", (PyCFunction)(void(*)(void))imperative_push_dense, METH_VARARGS | METH_KEYWORDS, "C++ interface function for push_dense in dygraph."},
  {"silu", (PyCFunction)(void(*)(void))imperative_silu, METH_VARARGS | METH_KEYWORDS, "C++ interface function for silu in dygraph."},
  {"sequence_erase", (PyCFunction)(void(*)(void))imperative_sequence_erase, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sequence_erase in dygraph."},
  {"real", (PyCFunction)(void(*)(void))imperative_real, METH_VARARGS | METH_KEYWORDS, "C++ interface function for real in dygraph."},
  {"nearest_interp_v2", (PyCFunction)(void(*)(void))imperative_nearest_interp_v2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for nearest_interp_v2 in dygraph."},
  {"dgc_clip_by_norm", (PyCFunction)(void(*)(void))imperative_dgc_clip_by_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dgc_clip_by_norm in dygraph."},
  {"squeeze2", (PyCFunction)(void(*)(void))imperative_squeeze2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze2 in dygraph."},
  {"squeeze2_", (PyCFunction)(void(*)(void))imperative_squeeze2_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for squeeze2_ in dygraph."},
  {"conj", (PyCFunction)(void(*)(void))imperative_conj, METH_VARARGS | METH_KEYWORDS, "C++ interface function for conj in dygraph."},
  {"strided_slice", (PyCFunction)(void(*)(void))imperative_strided_slice, METH_VARARGS | METH_KEYWORDS, "C++ interface function for strided_slice in dygraph."},
  {"precision_recall", (PyCFunction)(void(*)(void))imperative_precision_recall, METH_VARARGS | METH_KEYWORDS, "C++ interface function for precision_recall in dygraph."},
  {"fusion_seqexpand_concat_fc", (PyCFunction)(void(*)(void))imperative_fusion_seqexpand_concat_fc, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fusion_seqexpand_concat_fc in dygraph."},
  {"save", (PyCFunction)(void(*)(void))imperative_save, METH_VARARGS | METH_KEYWORDS, "C++ interface function for save in dygraph."},
  {"depthwise_conv2d_transpose", (PyCFunction)(void(*)(void))imperative_depthwise_conv2d_transpose, METH_VARARGS | METH_KEYWORDS, "C++ interface function for depthwise_conv2d_transpose in dygraph."},
  {"fake_quantize_range_abs_max", (PyCFunction)(void(*)(void))imperative_fake_quantize_range_abs_max, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fake_quantize_range_abs_max in dygraph."},
  {"positive_negative_pair", (PyCFunction)(void(*)(void))imperative_positive_negative_pair, METH_VARARGS | METH_KEYWORDS, "C++ interface function for positive_negative_pair in dygraph."},
  {"square", (PyCFunction)(void(*)(void))imperative_square, METH_VARARGS | METH_KEYWORDS, "C++ interface function for square in dygraph."},
  {"square_", (PyCFunction)(void(*)(void))imperative_square_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for square_ in dygraph."},
  {"var_conv_2d", (PyCFunction)(void(*)(void))imperative_var_conv_2d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for var_conv_2d in dygraph."},
  {"log1p", (PyCFunction)(void(*)(void))imperative_log1p, METH_VARARGS | METH_KEYWORDS, "C++ interface function for log1p in dygraph."},
  {"channel_shuffle", (PyCFunction)(void(*)(void))imperative_channel_shuffle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for channel_shuffle in dygraph."},
  {"sparse_indices", (PyCFunction)(void(*)(void))imperative_sparse_indices, METH_VARARGS | METH_KEYWORDS, "C++ interface function for sparse_indices in dygraph."},
  {"atan2", (PyCFunction)(void(*)(void))imperative_atan2, METH_VARARGS | METH_KEYWORDS, "C++ interface function for atan2 in dygraph."},
  {"fused_softmax_mask_upper_triangle", (PyCFunction)(void(*)(void))imperative_fused_softmax_mask_upper_triangle, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fused_softmax_mask_upper_triangle in dygraph."},
  {"clip_by_norm", (PyCFunction)(void(*)(void))imperative_clip_by_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for clip_by_norm in dygraph."},
  {"box_decoder_and_assign", (PyCFunction)(void(*)(void))imperative_box_decoder_and_assign, METH_VARARGS | METH_KEYWORDS, "C++ interface function for box_decoder_and_assign in dygraph."},
  {"roi_pool", (PyCFunction)(void(*)(void))imperative_roi_pool, METH_VARARGS | METH_KEYWORDS, "C++ interface function for roi_pool in dygraph."},
  {"fft_r2c", (PyCFunction)(void(*)(void))imperative_fft_r2c, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fft_r2c in dygraph."},
  {"overlap_add", (PyCFunction)(void(*)(void))imperative_overlap_add, METH_VARARGS | METH_KEYWORDS, "C++ interface function for overlap_add in dygraph."},
  {"fill_constant_batch_size_like", (PyCFunction)(void(*)(void))imperative_fill_constant_batch_size_like, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_constant_batch_size_like in dygraph."},
  {"fill_any", (PyCFunction)(void(*)(void))imperative_fill_any, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_any in dygraph."},
  {"fill_any_", (PyCFunction)(void(*)(void))imperative_fill_any_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for fill_any_ in dygraph."},
  {"dequantize_log", (PyCFunction)(void(*)(void))imperative_dequantize_log, METH_VARARGS | METH_KEYWORDS, "C++ interface function for dequantize_log in dygraph."},
  {"max_pool2d_with_index", (PyCFunction)(void(*)(void))imperative_max_pool2d_with_index, METH_VARARGS | METH_KEYWORDS, "C++ interface function for max_pool2d_with_index in dygraph."},
  {"pad3d", (PyCFunction)(void(*)(void))imperative_pad3d, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pad3d in dygraph."},
  {"norm", (PyCFunction)(void(*)(void))imperative_norm, METH_VARARGS | METH_KEYWORDS, "C++ interface function for norm in dygraph."},
  {"viterbi_decode", (PyCFunction)(void(*)(void))imperative_viterbi_decode, METH_VARARGS | METH_KEYWORDS, "C++ interface function for viterbi_decode in dygraph."},
  {"mish", (PyCFunction)(void(*)(void))imperative_mish, METH_VARARGS | METH_KEYWORDS, "C++ interface function for mish in dygraph."},
  {"box_coder", (PyCFunction)(void(*)(void))imperative_box_coder, METH_VARARGS | METH_KEYWORDS, "C++ interface function for box_coder in dygraph."},
  {"flatten", (PyCFunction)(void(*)(void))imperative_flatten, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten in dygraph."},
  {"flatten_", (PyCFunction)(void(*)(void))imperative_flatten_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for flatten_ in dygraph."},
  {"elementwise_mod", (PyCFunction)(void(*)(void))imperative_elementwise_mod, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_mod in dygraph."},
  {"elementwise_mod_", (PyCFunction)(void(*)(void))imperative_elementwise_mod_, METH_VARARGS | METH_KEYWORDS, "C++ interface function for elementwise_mod_ in dygraph."},
  {"margin_cross_entropy", (PyCFunction)(void(*)(void))imperative_margin_cross_entropy, METH_VARARGS | METH_KEYWORDS, "C++ interface function for margin_cross_entropy in dygraph."},
  {"pull_sparse", (PyCFunction)(void(*)(void))imperative_pull_sparse, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pull_sparse in dygraph."},
  {"logical_and", (PyCFunction)(void(*)(void))imperative_logical_and, METH_VARARGS | METH_KEYWORDS, "C++ interface function for logical_and in dygraph."},
  {"pow", (PyCFunction)(void(*)(void))imperative_pow, METH_VARARGS | METH_KEYWORDS, "C++ interface function for pow in dygraph."},
  {nullptr,nullptr,0,nullptr}};

void BindOpFunctions4(pybind11::module *module) {
  auto m = module->def_submodule("ops");
  if (PyModule_AddFunctions(m.ptr(), ExtestMethods) < 0) {
    PADDLE_THROW(platform::errors::Fatal ("Add functions to core.ops failed!"));
  }

  InitOpsAttrTypeMap();}

} // namespace pybind
} // namespace paddle
