%module(directors="1") swig_paddle
%include "std_string.i"
%{
#define SWIG_FILE_WITH_INIT
#include "api/PaddleAPI.h"   
%}

%include "exception.i"
%typemap(throws) UnsupportError %{
  SWIG_exception(SWIG_RuntimeError, $1.what());
  SWIG_fail;
%}

%include "std_vector.i"
%include "std_pair.i"
#ifdef SWIGPYTHON
%include "numpy.i"
#endif

%init %{
#ifdef SWIGPYTHON
import_array();
#endif
%}


namespace std {
%template(vector_int) vector<int>;
%template(vector_uint) vector<unsigned int>;
%template(vector_float) vector<float>;
%template(vector_string) vector<string>;
%template(vector_vec_star) vector<Vector*>;
}
#ifdef SWIGPYTHON 
%typemap(in) (int argc, char** argv) { 
    int i = 0; 
    if (!PyList_Check($input)) { 
        PyErr_SetString(PyExc_ValueError, "Expecting a list"); 
        return NULL; 
    } 
    $1 = PyList_Size($input); 
    $2 = (char **) malloc(($1+1)*sizeof(char *)); 
    for (i = 0; i < $1; i++) { 
        PyObject *s = PyList_GetItem($input,i); 
        if (!PyString_Check(s)) { 
            free($2); 
            PyErr_SetString(PyExc_ValueError, "List items must be strings"); 
            return NULL; 
        } 
        $2[i] = PyString_AsString(s); 
    } 
    $2[i] = 0; 
} 
%typemap(freearg) (int argc, char** argv) { 
    if ($2) free($2); 
} 

%typemap(out) FloatArray {
  $result = PyList_New($1.length);
  for (size_t i=0; i<$1.length; ++i) {
    PyList_SetItem($result, i, PyFloat_FromDouble($1.buf[i]));
  }  
  if($1.needFree) {
    delete [] $1.buf;  
  }
}

%typemap(out) IntArray {
  $result = PyList_New($1.length);  
  for (size_t i=0; i<$1.length; ++i) {
    PyList_SetItem($result, i, PyInt_FromLong($1.buf[i]));  
  }
  if ($1.needFree) {
    delete [] $1.buf;  
  }
}

%typemap(out) IntWithFloatArray {
  $result = PyList_New($1.length);
  for (size_t i=0; i<$1.length; ++i) {
    PyList_SetItem($result, i, PyTuple_Pack(2, 
      PyInt_FromLong($1.idxBuf[i]),
      PyFloat_FromDouble($1.valBuf[i])
    ));
  }
  if ($1.needFree) {
    delete [] $1.idxBuf;
    delete [] $1.valBuf;
  } 
}


%rename(__getitem__) IVector::get;
%rename(__setitem__) IVector::set;
%rename(__len__) IVector::getSize;
%rename(__getitem__) Vector::get;
%rename(__setitem__) Vector::set;
%rename(__len__) Vector::getSize;
%rename(__len__) Parameter::getSize;
%rename(__call__) ParameterTraverseCallback::apply;
%rename(__repr__) Evaluator::toString;

%apply (float* INPLACE_ARRAY2, int DIM1, int DIM2) { 
  (float* data, int dim1, int dim2) 
}

%apply (float** ARGOUTVIEW_ARRAY2, int* DIM1, int* DIM2) { 
  (float** view_data, int* dim1, int* dim2) 
}

%apply (float** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {
  (float** view_m_data, int* dim1, int* dim2)  
}

%apply (int** ARGOUTVIEWM_ARRAY1, int* DIM1) {
  (int** view_m_data, int* dim1)  
}

%apply (int* INPLACE_ARRAY1, int DIM1) { 
  (int* data, int dim) 
}

%apply (int** ARGOUTVIEW_ARRAY1, int* DIM1) {
  (int** view_data, int* dim1)  
}

%apply (float* INPLACE_ARRAY1, int DIM1) {
  (float* data, int dim)
}

%apply (float** ARGOUTVIEW_ARRAY1, int* DIM1) {
  (float** view_data, int* dim1)
}

%apply (float** ARGOUTVIEWM_ARRAY1, int* DIM1) {
  (float** view_m_data, int* dim1)
}

#endif
// The below functions internally create object by "new", so it should use
// use SWIG to handle gc. There are hints for SWIG to handle GC.
%newobject Matrix::createZero;
%newobject Matrix::createSparse;
%newobject Matrix::createDense;
%newobject Matrix::createDenseFromNumpy;
%newobject Matrix::createCpuDenseFromNumpy;
%newobject Matrix::createGpuDenseFromNumpy;
%newobject Vector::createZero;
%newobject Vector::create;
%newobject Vector::createVectorFromNumpy;
%newobject Vector::createCpuVectorFromNumpy;
%newobject Vector::createGpuVectorFromNumpy;
%newobject IVector::createZero;
%newobject IVector::create;
%newobject IVector::createVectorFromNumpy;
%newobject IVector::createCpuVectorFromNumpy;
%newobject IVector::createGpuVectorFromNumpy;
%newobject Trainer::createByCommandLine;
%newobject Trainer::getForwardOutput;
%newobject Trainer::getLayerOutput;
%newobject Arguments::getSlotValue;
%newobject Arguments::getSlotIds;
%newobject Arguments::getSlotIn;
%newobject Arguments::getSlotSequenceStartPositions;
%newobject Arguments::getSlotSequenceDim;
%newobject Arguments::createArguments;
%newobject GradientMachine::createByConfigProtoStr;
%newobject GradientMachine::createByModelConfig;
%newobject GradientMachine::asSequenceGenerator;
%newobject GradientMachine::getParameter;
%newobject GradientMachine::getLayerOutput;
%newobject GradientMachine::makeEvaluator;
%newobject TrainerConfig::createFromTrainerConfigFile;
%newobject TrainerConfig::getModelConfig;
%newobject TrainerConfig::getOptimizationConfig;
%newobject Parameter::getBuf;
%newobject Parameter::getConfig;
%newobject ParameterOptimizer::create;
%newobject ParameterOptimizer::needSpecialTraversal;
%newobject ParameterUpdater::createLocalUpdater;
%newobject ParameterUpdater::createRemoteUpdater;
%newobject ParameterUpdater::createNewRemoteUpdater;

%feature("director") UpdateCallback;
%feature("autodoc", 1); // To generate method stub, for code hint in ide

// Ignore many private class, and method cannot be handled by swig.
%ignore MatrixPrivate;
%ignore TrainerPrivate;
%ignore IVector::operator[];
%ignore ArgumentsPrivate;
%ignore GradientMachinePrivate;
%ignore TrainerConfigPrivate;
%ignore ModelConfigPrivate;
%ignore ParameterPrivate;
%ignore SequenceGeneratorPrivate;
%ignore VectorPrivate;
%ignore ParameterConfigPrivate;
%ignore OptimizationConfigPrivate;
%ignore ParameterTraverseCallbackPrivate;
%include "utils/GlobalConstants.h"
%include "api/PaddleAPI.h"
