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
  if(WIN32)
    # Link the import library generated from libpaddle.pyd. The paddle target's
    # import library names libpaddle.dll, which is not present in the Python
    # test layout.
    target_link_libraries(
      scalar_type_compat_test
      PRIVATE "${PADDLE_BINARY_DIR}/python/paddle/base/libpaddle.lib"
              ${PYTHON_LIBRARIES})
    add_dependencies(scalar_type_compat_test copy_libpaddle)
  else()
    target_link_libraries(scalar_type_compat_test
                          PRIVATE $<TARGET_LINKER_FILE:${paddle_lib}>)
  endif()
  if(APPLE)
    add_dependencies(scalar_type_compat_test copy_libpaddle)
    target_link_libraries(
      scalar_type_compat_test PRIVATE "-Wl,-undefined,dynamic_lookup"
                                      "-Wl,-rpath,@loader_path")
    # Match Paddle's macOS custom-extension layout: resolve libpaddle.dylib to
    # the libpaddle.so that Python has already loaded, rather than loading the
    # build-tree dylib a second time and registering Paddle flags twice.
    add_custom_command(
      TARGET scalar_type_compat_test
      POST_BUILD
      COMMAND
        ${CMAKE_COMMAND} -E create_symlink
        "../../python/paddle/base/libpaddle.so"
        "$<TARGET_FILE_DIR:scalar_type_compat_test>/libpaddle.dylib"
      VERBATIM)
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
