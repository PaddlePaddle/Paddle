if(NOT WITH_CPP_TEST)
  return()
endif()
add_subdirectory(auto_parallel)
add_subdirectory(phi)
add_subdirectory(jit)
add_subdirectory(new_executor)
add_subdirectory(prim)
add_subdirectory(imperative)
add_subdirectory(pir)
add_subdirectory(inference)
add_subdirectory(eager)
add_subdirectory(fluid)
add_subdirectory(utils)
add_subdirectory(compat)
if(WITH_CINN)
  add_subdirectory(cinn)
endif()

if(WIN32 AND WITH_ONNXRUNTIME)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_SHARED_LIB}"
            "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll"
    DEPENDS onnxruntime
    COMMENT "Copying onnxruntime.dll to build/test/cpp")

  add_custom_target(copy_onnxruntime ALL
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime.dll)
endif()
