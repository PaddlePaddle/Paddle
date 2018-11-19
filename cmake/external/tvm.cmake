include(ExternalProject)

set(TVM_PROJECT       "extern_tvm")
set(TVM_SOURCES_DIR   ${THIRD_PARTY_PATH}/tvm)
set(TVM_INSTALL_DIR   ${THIRD_PARTY_PATH}/tvm/src/extern_tvm-build)
include_directories(${TVM_SOURCES_DIR}/src/extern_tvm/include)
include_directories(${TVM_SOURCES_DIR}/src/extern_tvm/dmlc-core/include)
include_directories(${TVM_SOURCES_DIR}/src/extern_tvm/dlpack/include)
include_directories(${TVM_SOURCES_DIR}/src/extern_tvm/HalideIR/src)

if (WITH_GPU)
    set(TVM_BUILD_OPTIONS -DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_CUBLAS=ON)
else()
    set(TVM_BUILD_OPTIONS -DUSE_CUDA=OFF -DUSE_CUDNN=OFF -DUSE_CUBLAS=OFF)
endif()

list(APPEND TVM_BUILD_OPTIONS -DUSE_LLVM=${LLVM_BUILD_DIR}/bin/llvm-config)

ExternalProject_Add(
    ${TVM_PROJECT}
    ${EXTERNAL_PROJECT_LOG_ARGS}
    GIT_REPOSITORY        "https://github.com/dmlc/tvm.git"
    GIT_TAG               "v0.4"
    PREFIX                ${TVM_SOURCES_DIR}
    DEPENDS               ${LLVM_PROJECT}
    CMAKE_ARGS            -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    CMAKE_ARGS            ${TVM_BUILD_OPTIONS} 
)

add_library(shared_tvm_core SHARED IMPORTED GLOBAL)
set_property(TARGET shared_tvm_core PROPERTY IMPORTED_LOCATION ${TVM_INSTALL_DIR}/libtvm.so)
add_dependencies(shared_tvm_core ${TVM_PROJECT})

add_library(shared_tvm_runtime SHARED IMPORTED GLOBAL)
set_property(TARGET shared_tvm_runtime PROPERTY IMPORTED_LOCATION ${TVM_INSTALL_DIR}/libtvm_runtime.so)
add_dependencies(shared_tvm_runtime ${TVM_PROJECT})

add_library(shared_tvm_topi SHARED IMPORTED GLOBAL)
set_property(TARGET shared_tvm_topi PROPERTY IMPORTED_LOCATION ${TVM_INSTALL_DIR}/libtvm_topi.so)
add_dependencies(shared_tvm_topi ${TVM_PROJECT})

add_library(shared_tvm_vta SHARED IMPORTED GLOBAL)
set_property(TARGET shared_tvm_vta PROPERTY IMPORTED_LOCATION ${TVM_INSTALL_DIR}/libvta.so)
add_dependencies(shared_tvm_vta ${TVM_PROJECT})

add_library(shared_tvm_nnvm_compiler SHARED IMPORTED GLOBAL)
set_property(TARGET shared_tvm_nnvm_compiler PROPERTY IMPORTED_LOCATION ${TVM_INSTALL_DIR}/libnnvm_compiler.so)
add_dependencies(shared_tvm_nnvm_compiler ${TVM_PROJECT})

add_custom_target(shared_tvm DEPENDS shared_tvm_core shared_tvm_runtime shared_tvm_topi shared_tvm_vta shared_tvm_nnvm_compiler)

list(APPEND external_project_dependencies shared_tvm)

set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/tvm_dummy.c)
file(WRITE ${dummyfile} "const char * dummy = \"${dummyfile}\";")
add_library(tvm STATIC ${dummyfile})
target_link_libraries(tvm ${TVM_INSTALL_DIR}/libnnvm_compiler.so)
target_link_libraries(tvm ${TVM_INSTALL_DIR}/libtvm.so)
target_link_libraries(tvm ${TVM_INSTALL_DIR}/libtvm_topi.so)
target_link_libraries(tvm ${TVM_INSTALL_DIR}/libvta.so)
target_link_libraries(tvm ${TVM_INSTALL_DIR}/libtvm_runtime.so)
add_dependencies(tvm ${TVM_PROJECT})

if (WITH_C_API)
    install(FILES ${TVM_INSTALL_DIR}/libtvm.so DESTINATION lib)
    install(FILES ${TVM_INSTALL_DIR}/libtvm_runtime.so DESTINATION lib)
    install(FILES ${TVM_INSTALL_DIR}/libtvm_topi.so DESTINATION lib)
    install(FILES ${TVM_INSTALL_DIR}/libvta.so DESTINATION lib)
    install(FILES ${TVM_INSTALL_DIR}/libnnvm_compiler.so DESTINATION lib)
endif()
