if(NOT WITH_ONNXRUNTIME)
  return()
endif()

set(PADDLE2ONNX_ROOT "/usr" CACHE PATH "ONNXRUNTIME ROOT")
# SET(PADDLE2ONNX_PROJECT           "extern_paddle2onnx")

message("paddle2onnx --------CMakeLists-------:${PADDLE2ONNX_ROOT}")

# include_directories(/usr/local/include)
# include_directories(${PADDLE2ONNX_ROOT}/include)
# include_directories(${PADDLE2ONNX_ROOT}/include)
# link_directories(${PADDLE2ONNX_ROOT}/lib/libp2o_lib.so)

# ADD_LIBRARY(paddle2onnx SHARED IMPORTED GLOBAL)
# SET_PROPERTY(TARGET paddle2onnx PROPERTY IMPORTED_LOCATION ${PADDLE2ONNX_ROOT}/lib)
# ADD_DEPENDENCIES(paddle2onnx ${PADDLE2ONNX_PROJECT})
