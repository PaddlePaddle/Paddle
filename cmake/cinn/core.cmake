set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -fPIC -mavx -mfma -Wno-write-strings -Wno-psabi")

set(PADDLE_RESOURCE_URL
    "http://paddle-inference-dist.bj.bcebos.com"
    CACHE STRING "inference download url")

function(cinn_cc_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cinn_cc_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  if(cinn_cc_library_SRCS)
    if(cinn_cc_library_SHARED OR cinn_cc_library_shared) # build *.so
      add_library(${TARGET_NAME} SHARED ${cinn_cc_library_SRCS})
    else()
      add_library(${TARGET_NAME} STATIC ${cinn_cc_library_SRCS})
    endif()

    if(cinn_cc_library_DEPS)
      if("${cinn_cc_library_DEPS};" MATCHES "python;")
        list(REMOVE_ITEM cinn_cc_library_DEPS python)
        add_dependencies(${TARGET_NAME} python)
      endif()
      target_link_libraries(${TARGET_NAME} ${cinn_cc_library_DEPS})
      add_dependencies(${TARGET_NAME} ${cinn_cc_library_DEPS})
    endif()

    # cpplint code style
    foreach(source_file ${cinn_cc_library_SRCS})
      string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        list(APPEND cinn_cc_library_HEADERS
             ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
      endif()
    endforeach()
  else()
    if(cinn_cc_library_DEPS)
      cinn_merge_static_libs(${TARGET_NAME} ${cinn_cc_library_DEPS})
    else()
      message(
        FATAL_ERROR
          "Please specify source files or libraries in cinn_cc_library(${TARGET_NAME} ...)."
      )
    endif()
  endif()

  if((NOT ("${TARGET_NAME}" STREQUAL "cinn_gtest_main"))
     AND (NOT ("${TARGET_NAME}" STREQUAL "utils"))
     AND (NOT ("${TARGET_NAME}" STREQUAL "lib")))
    target_link_libraries(${TARGET_NAME} Threads::Threads)

  endif(
    (NOT ("${TARGET_NAME}" STREQUAL "cinn_gtest_main"))
    AND (NOT ("${TARGET_NAME}" STREQUAL "utils"))
    AND (NOT ("${TARGET_NAME}" STREQUAL "lib")))
endfunction()

list(APPEND CMAKE_CTEST_ARGUMENTS)

function(remove_gflags TARGET_NAME)
  get_target_property(TARGET_LIBRARIES ${TARGET_NAME} LINK_LIBRARIES)
  list(REMOVE_ITEM TARGET_LIBRARIES glog)
  list(REMOVE_ITEM TARGET_LIBRARIES gflags)
  set_property(TARGET ${TARGET_NAME} PROPERTY LINK_LIBRARIES
                                              ${TARGET_LIBRARIES})
endfunction()

function(cinn_cc_test TARGET_NAME)
  if(WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS)
    cmake_parse_arguments(cinn_cc_test "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET_NAME} ${cinn_cc_test_SRCS})
    get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
    target_link_libraries(${TARGET_NAME} ${os_dependency_modules}
                          cinn_gtest_main gtest glog ${cinn_cc_test_DEPS})
    add_dependencies(${TARGET_NAME} cinn_gtest_main gtest glog
                     ${cinn_cc_test_DEPS})

    add_test(
      NAME ${TARGET_NAME}
      COMMAND ${TARGET_NAME} "${cinn_cc_test_ARGS}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if(${cinn_cc_test_SERIAL})
      set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
    # No unit test should exceed 10 minutes.
    set_tests_properties(${TARGET_NAME} PROPERTIES TIMEOUT 6000)
    remove_gflags(${TARGET_NAME})
  endif()
endfunction()

function(cinn_nv_library TARGET_NAME)
  if(WITH_GPU)
    set(options STATIC static SHARED shared)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(cinn_nv_library "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})
    if(cinn_nv_library_SRCS)
      if(cinn_nv_library_SHARED OR cinn_nv_library_shared) # build *.so
        cuda_add_library(${TARGET_NAME} SHARED ${cinn_nv_library_SRCS})
      else()
        cuda_add_library(${TARGET_NAME} STATIC ${cinn_nv_library_SRCS})
      endif()
      if(cinn_nv_library_DEPS)
        add_dependencies(${TARGET_NAME} ${cinn_nv_library_DEPS})
        target_link_libraries(${TARGET_NAME} ${cinn_nv_library_DEPS})
      endif()
      # cpplint code style
      foreach(source_file ${cinn_nv_library_SRCS})
        string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
          list(APPEND cinn_nv_library_HEADERS
               ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        endif()
      endforeach()
    else()
      if(cinn_nv_library_DEPS)
        cinn_merge_static_libs(${TARGET_NAME} ${cinn_nv_library_DEPS})
      else()
        message(FATAL
                "Please specify source file or library in cinn_nv_library.")
      endif()
    endif()
    target_link_libraries(${TARGET_NAME} Threads::Threads)
  endif()
endfunction()

function(cinn_nv_binary TARGET_NAME)
  if(WITH_GPU)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(cinn_nv_binary "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})
    cuda_add_executable(${TARGET_NAME} ${cinn_nv_binary_SRCS})
    if(cinn_nv_binary_DEPS)
      target_link_libraries(${TARGET_NAME} ${cinn_nv_binary_DEPS})
      add_dependencies(${TARGET_NAME} ${cinn_nv_binary_DEPS})
      common_link(${TARGET_NAME})
    endif()
  endif()
endfunction()

function(cinn_nv_test TARGET_NAME)
  if(WITH_GPU AND WITH_TESTING)
    set(options SERIAL)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS)
    cmake_parse_arguments(cinn_nv_test "${options}" "${oneValueArgs}"
                          "${multiValueArgs}" ${ARGN})
    # Attention:
    # 1. cuda_add_executable is deprecated after cmake v3.10, use cuda_add_executable for CUDA please.
    # 2. cuda_add_executable does not support ccache.
    # Reference: https://cmake.org/cmake/help/v3.10/module/FindCUDA.html
    add_executable(${TARGET_NAME} ${cinn_nv_test_SRCS})
    get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
    target_link_libraries(
      ${TARGET_NAME}
      ${cinn_nv_test_DEPS}
      cinn_gtest_main
      gtest
      ${os_dependency_modules}
      ${CUDNN_LIBRARY}
      ${CUBLAS_LIBRARIES}
      ${CUDA_LIBRARIES})
    add_dependencies(${TARGET_NAME} ${cinn_nv_test_DEPS} cinn_gtest_main gtest)
    common_link(${TARGET_NAME})
    add_test(
      NAME ${TARGET_NAME}
      COMMAND ${TARGET_NAME} "${cinn_nv_test_ARGS}"
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    if(cinn_nv_test_SERIAL)
      set_property(TEST ${TARGET_NAME} PROPERTY RUN_SERIAL 1)
    endif()
    target_link_libraries(
      ${TARGET_NAME} Threads::Threads ${CUDA_NVRTC_LIB} ${CUDA_LIBRARIES}
      ${CUDA_cudart_static_LIBRARY}
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs/libcuda.so)
    if(NVTX_FOUND)
      target_link_libraries(${TARGET_NAME} ${CUDA_NVTX_LIB})
    endif()
    remove_gflags(${TARGET_NAME})
  endif()
endfunction()

# Add dependency that TARGET will depend on test result of DEP, this function executes the DEP during make.
function(add_run_test_dependency TARGET_NAME DEP_NAME)
  if(WITH_TESTING)
    set(custom_target_name ${TARGET_NAME}_TEST_OUTPUT_DEPENDENCY_ON_${DEP_NAME})
    add_custom_target(
      ${custom_target_name}
      COMMAND
        cd ${CMAKE_CURRENT_BINARY_DIR} && ./${DEP_NAME}
        --cinn_x86_builtin_code_root=${CMAKE_SOURCE_DIR}/paddle/cinn/backends
      COMMAND cd ${CMAKE_BINARY_DIR}
      DEPENDS ${DEP_NAME})
    add_dependencies(${TARGET_NAME} ${DEP_NAME} ${custom_target_name})
  endif()
endfunction()

# find all third_party modules is used for paddle static library
# for reduce the dependency when building the inference libs.
set_property(GLOBAL PROPERTY FLUID_THIRD_PARTY)
function(find_fluid_thirdparties TARGET_NAME)
  get_filename_component(__target_path ${TARGET_NAME} ABSOLUTE)
  string(REGEX REPLACE "^${PADDLE_SOURCE_DIR}/" "" __target_path
                       ${__target_path})
  string(FIND "${__target_path}" "third_party" pos)
  if(pos GREATER 1)
    get_property(fluid_ GLOBAL PROPERTY FLUID_THIRD_PARTY)
    set(fluid_third_partys ${fluid_third_partys} ${TARGET_NAME})
    set_property(GLOBAL PROPERTY FLUID_THIRD_PARTY "${fluid_third_partys}")
  endif()
endfunction()

function(cinn_merge_static_libs TARGET_NAME)
  set(libs ${ARGN})
  list(REMOVE_DUPLICATES libs)

  # Get all propagation dependencies from the merged libraries
  foreach(lib ${libs})
    list(APPEND libs_deps ${${lib}_LIB_DEPENDS})
  endforeach()
  if(libs_deps)
    list(REMOVE_DUPLICATES libs_deps)
  endif()

  # To produce a library we need at least one source file.
  # It is created by add_custom_command below and will helps
  # also help to track dependencies.
  set(target_SRCS ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}_dummy.c)

  if(APPLE) # Use OSX's libtool to merge archives
    # Make the generated dummy source file depended on all static input
    # libs. If input lib changes,the source file is touched
    # which causes the desired effect (relink).
    add_custom_command(
      OUTPUT ${target_SRCS}
      COMMAND ${CMAKE_COMMAND} -E touch ${target_SRCS}
      DEPENDS ${libs})

    # Generate dummy static lib
    file(WRITE ${target_SRCS}
         "const char *dummy_${TARGET_NAME} = \"${target_SRCS}\";")
    add_library(${TARGET_NAME} STATIC ${target_SRCS})
    target_link_libraries(${TARGET_NAME} ${libs_deps})

    foreach(lib ${libs})
      # Get the file names of the libraries to be merged
      set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
    endforeach()
    add_custom_command(
      TARGET ${TARGET_NAME}
      POST_BUILD
      COMMAND rm "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.a"
      COMMAND /usr/bin/libtool -static -o
              "${CMAKE_CURRENT_BINARY_DIR}/lib${TARGET_NAME}.a" ${libfiles})
  endif()
  if(LINUX
  )# general UNIX: use "ar" to extract objects and re-add to a common lib
    set(target_DIR ${CMAKE_CURRENT_BINARY_DIR}/${TARGET_NAME}.dir)

    foreach(lib ${libs})
      set(objlistfile ${target_DIR}/${lib}.objlist
      )# list of objects in the input library
      set(objdir ${target_DIR}/${lib}.objdir)

      add_custom_command(
        OUTPUT ${objdir}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${objdir}
        DEPENDS ${lib})

      add_custom_command(
        OUTPUT ${objlistfile}
        COMMAND ${CMAKE_AR} -x "$<TARGET_FILE:${lib}>"
        COMMAND ${CMAKE_AR} -t "$<TARGET_FILE:${lib}>" > ${objlistfile}
        DEPENDS ${lib} ${objdir}
        WORKING_DIRECTORY ${objdir})

      list(APPEND target_OBJS "${objlistfile}")
    endforeach()

    # Make the generated dummy source file depended on all static input
    # libs. If input lib changes,the source file is touched
    # which causes the desired effect (relink).
    add_custom_command(
      OUTPUT ${target_SRCS}
      COMMAND ${CMAKE_COMMAND} -E touch ${target_SRCS}
      DEPENDS ${libs} ${target_OBJS})

    # Generate dummy static lib
    file(WRITE ${target_SRCS}
         "const char *dummy_${TARGET_NAME} = \"${target_SRCS}\";")
    add_library(${TARGET_NAME} STATIC ${target_SRCS})
    target_link_libraries(${TARGET_NAME} ${libs_deps})

    # Get the file name of the generated library
    set(target_LIBNAME "$<TARGET_FILE:${TARGET_NAME}>")

    add_custom_command(
      TARGET ${TARGET_NAME}
      POST_BUILD
      COMMAND ${CMAKE_AR} crs ${target_LIBNAME} `find ${target_DIR} -name '*.o'`
      COMMAND ${CMAKE_RANLIB} ${target_LIBNAME}
      WORKING_DIRECTORY ${target_DIR})
  endif()
  if(WIN32)

    # windows do not support gcc/nvcc combined compiling. Use msvc lib.exe to merge libs.
    # Make the generated dummy source file depended on all static input
    # libs. If input lib changes,the source file is touched
    # which causes the desired effect (relink).
    add_custom_command(
      OUTPUT ${target_SRCS}
      COMMAND ${CMAKE_COMMAND} -E touch ${target_SRCS}
      DEPENDS ${libs})

    # Generate dummy static lib
    file(WRITE ${target_SRCS}
         "const char *dummy_${TARGET_NAME} = \"${target_SRCS}\";")
    add_library(${TARGET_NAME} STATIC ${target_SRCS})
    target_link_libraries(${TARGET_NAME} ${libs_deps})

    foreach(lib ${libs})
      # Get the file names of the libraries to be merged
      set(libfiles ${libfiles} $<TARGET_FILE:${lib}>)
    endforeach()
    # msvc will put library in directory of "/Release/xxxlib" by default
    #       COMMAND cmake -E remove "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/${TARGET_NAME}.lib"
    add_custom_command(
      TARGET ${TARGET_NAME}
      POST_BUILD
      COMMAND cmake -E make_directory
              "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}"
      COMMAND
        lib
        /OUT:${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib${TARGET_NAME}.lib
        ${libfiles})
  endif()
endfunction()

# Modification of standard 'protobuf_generate_cpp()' with protobuf-lite support
# Usage:
#   paddle_protobuf_generate_cpp(<proto_srcs> <proto_hdrs> <proto_files>)

function(paddle_protobuf_generate_cpp SRCS HDRS)
  if(NOT ARGN)
    message(
      SEND_ERROR
        "Error: paddle_protobuf_generate_cpp() called without any proto files")
    return()
  endif()

  set(${SRCS})
  set(${HDRS})

  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)

    set(_protobuf_protoc_src "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    set(_protobuf_protoc_hdr "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")
    list(APPEND ${SRCS} "${_protobuf_protoc_src}")
    list(APPEND ${HDRS} "${_protobuf_protoc_hdr}")

    add_custom_command(
      OUTPUT "${_protobuf_protoc_src}" "${_protobuf_protoc_hdr}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE} -I${CMAKE_SOURCE_DIR} --cpp_out
              "${CMAKE_BINARY_DIR}" ${ABS_FIL}
      DEPENDS ${ABS_FIL} protoc
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM)
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS}
      ${${SRCS}}
      PARENT_SCOPE)
  set(${HDRS}
      ${${HDRS}}
      PARENT_SCOPE)
endfunction()

function(cinn_proto_library TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cinn_proto_library "${options}" "${oneValueArgs}"
                        "${multiValueArgs}" ${ARGN})
  set(proto_srcs)
  set(proto_hdrs)
  paddle_protobuf_generate_cpp(proto_srcs proto_hdrs ${cinn_proto_library_SRCS})
  cinn_cc_library(${TARGET_NAME} SRCS ${proto_srcs} DEPS
                  ${cinn_proto_library_DEPS} protobuf)
  set("${TARGET_NAME}_HDRS"
      ${proto_hdrs}
      PARENT_SCOPE)
  set("${TARGET_NAME}_SRCS"
      ${proto_srcs}
      PARENT_SCOPE)
endfunction()

function(common_link TARGET_NAME)
  if(WITH_PROFILER)
    target_link_libraries(${TARGET_NAME} gperftools::profiler)
  endif()

  if(WITH_JEMALLOC)
    target_link_libraries(${TARGET_NAME} jemalloc::jemalloc)
  endif()
endfunction()

# This method is borrowed from Paddle-Lite.
function(download_and_uncompress INSTALL_DIR URL FILENAME)
  message(STATUS "Download inference test stuff from ${URL}/${FILENAME}")
  string(REGEX REPLACE "[-%.]" "_" FILENAME_EX ${FILENAME})
  set(EXTERNAL_PROJECT_NAME "extern_lite_download_${FILENAME_EX}")
  set(UNPACK_DIR "${INSTALL_DIR}/src/${EXTERNAL_PROJECT_NAME}")
  ExternalProject_Add(
    ${EXTERNAL_PROJECT_NAME}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    PREFIX ${INSTALL_DIR}
    DOWNLOAD_COMMAND
      wget --no-check-certificate -q -O ${INSTALL_DIR}/${FILENAME}
      ${URL}/${FILENAME} && ${CMAKE_COMMAND} -E tar xzf
      ${INSTALL_DIR}/${FILENAME}
    DOWNLOAD_DIR ${INSTALL_DIR}
    DOWNLOAD_NO_PROGRESS 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    UPDATE_COMMAND ""
    INSTALL_COMMAND "")
endfunction()

function(gather_srcs SRC_GROUP)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs "SRCS")
  cmake_parse_arguments(prefix "" "" "${multiValueArgs}" ${ARGN})
  foreach(cpp ${prefix_SRCS})
    set(${SRC_GROUP}
        "${${SRC_GROUP}};${CMAKE_CURRENT_SOURCE_DIR}/${cpp}"
        CACHE INTERNAL "")
  endforeach()
endfunction()

function(core_gather_headers)
  file(
    GLOB includes
    LIST_DIRECTORIES false
    RELATIVE ${CMAKE_SOURCE_DIR}
    *.h)

  foreach(header ${includes})
    set(core_includes
        "${core_includes};${header}"
        CACHE INTERNAL "")
  endforeach()
endfunction()
