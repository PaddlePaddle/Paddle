file(
  GLOB TEST_OPS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "test_*.py")
string(REPLACE ".py" "" TEST_OPS "${TEST_OPS}")

if(WITH_TESTING AND WITH_PYTHON)
  get_property(paddle_lib GLOBAL PROPERTY PADDLE_LIB_NAME)
  add_library(scalar_type_compat_test MODULE scalar_type_extension.cc)
  target_compile_features(scalar_type_compat_test PRIVATE cxx_std_17)
  target_compile_definitions(
    scalar_type_compat_test
    PRIVATE TORCH_EXTENSION_NAME=scalar_type_compat_test)
  # The Windows OPENBLAS test build enables ON_INFER globally, but this module
  # must exercise the Python-only ScalarType caster.
  if(ON_INFER)
    if(MSVC)
      target_compile_options(scalar_type_compat_test
                             PRIVATE /UPADDLE_ON_INFERENCE)
    else()
      target_compile_options(scalar_type_compat_test
                             PRIVATE -UPADDLE_ON_INFERENCE)
    endif()
  endif()
  target_link_libraries(scalar_type_compat_test
                        PRIVATE $<TARGET_LINKER_FILE:${paddle_lib}>)
  if(WIN32)
    target_link_libraries(scalar_type_compat_test PRIVATE ${PYTHON_LIBRARIES})
  elseif(APPLE)
    target_link_libraries(scalar_type_compat_test
                          PRIVATE "-Wl,-undefined,dynamic_lookup")
  endif()
  add_dependencies(scalar_type_compat_test ${paddle_lib} extern_pybind)
  set_target_properties(
    scalar_type_compat_test
    PROPERTIES PREFIX ""
               LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
               RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
  if(WIN32)
    set_target_properties(scalar_type_compat_test PROPERTIES SUFFIX ".pyd")
  endif()
  foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${CONFIG} CONFIG_UPPER)
    set_target_properties(
      scalar_type_compat_test
      PROPERTIES LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER}
                 "${CMAKE_CURRENT_BINARY_DIR}"
                 RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER}
                 "${CMAKE_CURRENT_BINARY_DIR}")
  endforeach()
endif()

foreach(TEST_OP ${TEST_OPS})
  py_test_modules(${TEST_OP} MODULES ${TEST_OP})
endforeach()
