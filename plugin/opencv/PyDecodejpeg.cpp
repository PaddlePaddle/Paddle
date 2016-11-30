/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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
#include <time.h>
#include <vector>
#include <sys/time.h>
#include <unistd.h>
#include <glog/logging.h>
#include <numpy/arrayobject.h>
#include <boost/python.hpp>

#include "DataTransformer.h"

using namespace boost::python;

/**
 * DecodeJpeg is an image processing API for interfacing Python and C++
 * code DataTransformer, which used OpenCV and multi-threads to accelerate
 * image processing.
 * The Boost Python Library is used to wrap C++ interfaces.
 */

class DecodeJpeg {
public:
  /**
   * The constructor will create and initialize an object of DataTransformer.
   */
  DecodeJpeg(int threadNum,
             int capacity,
             bool isTest,
             bool isColor,
             int resize_min_size,
             int cropSizeH,
             int cropSizeW,
             PyObject* meanValues) {
    int channel = isColor ? 3 : 1;
    bool isEltMean = false;
    bool isChannelMean = false;
    float* mean = NULL;
    if (meanValues || meanValues != Py_None) {
      if (!PyArray_Check(meanValues)) {
        LOG(FATAL) << "Object is not a numpy array";
      }
      pyTypeCheck(meanValues);
      int size = PyArray_SIZE(reinterpret_cast<PyArrayObject*>(meanValues));
      isChannelMean = (size == channel) ? true : false;
      isEltMean = (size == channel * cropSizeH * cropSizeW) ? true : false;
      CHECK(isChannelMean != isEltMean);
      mean = (float*)PyArray_DATA(reinterpret_cast<PyArrayObject*>(meanValues));
    }
    tfhandlerPtr_ = std::make_shared<DataTransformer>(threadNum,
                                                      capacity,
                                                      isTest,
                                                      isColor,
                                                      cropSizeH,
                                                      cropSizeW,
                                                      resize_min_size,
                                                      isEltMean,
                                                      isChannelMean,
                                                      mean);
  }

  ~DecodeJpeg() {}

  /**
   * @brief This function is used to parse the Python object and convert
   *        the data to C++ format. Then it called the function of
   *        DataTransformer to start image processing.
   * @param pysrc    The input image list with string type.
   * @param pylabel  The input label of image.
   *                 It's type is numpy.array with int32.
   */
  void start(boost::python::list& pysrc, PyObject* pydlen, PyObject* pylabel) {
    vector<char*> data;
    int num = len(pysrc);
    for (int t = 0; t < num; ++t) {
      char* src = boost::python::extract<char*>(pysrc[t]);
      data.push_back(src);
    }
    int* dlen = (int*)PyArray_DATA(reinterpret_cast<PyArrayObject*>(pydlen));
    int* dlabels =
        (int*)PyArray_DATA(reinterpret_cast<PyArrayObject*>(pylabel));
    tfhandlerPtr_->start(data, dlen, dlabels);
  }

  /**
   * @brief Return one processed data.
   * @param pytrg    The processed image.
   * @param pylabel  The label of processed image.
   */
  void get(PyObject* pytrg, PyObject* pylab) {
    pyWritableCheck(pytrg);
    pyWritableCheck(pylab);
    pyContinuousCheck(pytrg);
    pyContinuousCheck(pylab);
    float* data = (float*)PyArray_DATA(reinterpret_cast<PyArrayObject*>(pytrg));
    int* label = (int*)PyArray_DATA(reinterpret_cast<PyArrayObject*>(pylab));
    tfhandlerPtr_->obtain(data, label);
  }

  /**
   * @brief An object of DataTransformer, which is used to call
   *        the image processing funtions.
   */
  std::shared_ptr<DataTransformer> tfhandlerPtr_;

private:
  /**
   * @brief Check whether the type of PyObject is valid or not.
   */
  void pyTypeCheck(PyObject* o) {
    int typenum = PyArray_TYPE(reinterpret_cast<PyArrayObject*>(o));

    // clang-format off
    int type =
        typenum == NPY_UBYTE ? CV_8U :
        typenum == NPY_BYTE ? CV_8S :
        typenum == NPY_USHORT ? CV_16U :
        typenum == NPY_SHORT ? CV_16S :
        typenum == NPY_INT || typenum == NPY_LONG ? CV_32S :
        typenum == NPY_FLOAT ? CV_32F :
        typenum == NPY_DOUBLE ? CV_64F : -1;
    // clang-format on

    if (type < 0) {
      LOG(FATAL) << "toMat: Data type = " << type << " is not supported";
    }
  }

  /**
   * @brief Check whether the PyObject is writable or not.
   */
  void pyWritableCheck(PyObject* o) {
    CHECK(PyArray_ISWRITEABLE(reinterpret_cast<PyArrayObject*>(o)));
  }

  /**
   * @brief Check whether the PyObject is c-contiguous or not.
   */
  void pyContinuousCheck(PyObject* o) {
    CHECK(PyArray_IS_C_CONTIGUOUS(reinterpret_cast<PyArrayObject*>(o)));
  }
};  // DecodeJpeg

/**
 * @brief Initialize the Python interpreter and numpy.
 */
static void initPython() {
  Py_Initialize();
  PyOS_sighandler_t sighandler = PyOS_getsig(SIGINT);
  import_array();
  PyOS_setsig(SIGINT, sighandler);
}

/**
 * Use Boost.Python to expose C++ interface to Python.
 */
BOOST_PYTHON_MODULE(DeJpeg) {
  initPython();
  class_<DecodeJpeg>("DecodeJpeg",
                     init<int, int, bool, bool, int, int, int, PyObject*>())
      .def("start", &DecodeJpeg::start)
      .def("get", &DecodeJpeg::get);
};
