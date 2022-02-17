if(NOT WITH_ONNXRUNTIME)
  return()
endif()

set(ONNXRUNTIME_ROOT "/usr" CACHE PATH "ONNXRUNTIME ROOT")
set(ONNXRUNTIME_LIB   ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so)


message("onnx2runtime --------CMakeLists-------${ONNXRUNTIME_ROOT}================${ONNXRUNTIME_LIB}")

add_definitions(-DPADDLE_WITH_ONNXRUNTIME)

include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)



ADD_LIBRARY(onnxruntime SHARED IMPORTED GLOBAL)
SET_PROPERTY(TARGET onnxruntime PROPERTY IMPORTED_LOCATION ${ONNXRUNTIME_LIB})
ADD_DEPENDENCIES(onnxruntime "extern_onnxruntmie")

