include(ExternalProject)

set(TVM_PROJECT       "extern_tvm")
set(TVM_SOURCES_DIR   ${THIRD_PARTY_PATH}/tvm)
set(TVM_INSTALL_DIR   ${THIRD_PARTY_PATH}/tvm/src/extern_tvm-build)
include_directories(${TVM_SOURCES_DIR}/src/extern_tvm/include)
include_directories(${TVM_SOURCES_DIR}/src/extern_tvm/3rdparty/dmlc-core/include)
include_directories(${TVM_SOURCES_DIR}/src/extern_tvm/3rdparty/dlpack/include)
include_directories(${TVM_SOURCES_DIR}/src/extern_tvm/3rdparty/HalideIR/src)

set(TVM_LIB "")
#list(APPEND TVM_LIB "${TVM_INSTALL_DIR}/libtvm_runtime.so")
list(APPEND TVM_LIB "${TVM_INSTALL_DIR}/libnnvm_compiler.so")
list(APPEND TVM_LIB "${TVM_INSTALL_DIR}/libtvm.so")
list(APPEND TVM_LIB "${TVM_INSTALL_DIR}/libtvm_topi.so")
list(APPEND TVM_LIB "${TVM_INSTALL_DIR}/libvta.so")

if (WITH_GPU)
    set(TVM_BUILD_OPTIONS -DUSE_CUDA=ON -DUSE_CUDNN=ON -DUSE_CUBLAS=ON)
else()
    set(TVM_BUILD_OPTIONS -DUSE_CUDA=OFF -DUSE_CUDNN=OFF -DUSE_CUBLAS=OFF)
endif()

list(APPEND TVM_BUILD_OPTIONS -DUSE_LLVM=${LLVM_BUILD_DIR}/bin/llvm-config)

ExternalProject_Add(
  	${TVM_PROJECT}
  	${EXTERNAL_PROJECT_LOG_ARGS}
  	GIT_REPOSITORY  		"https://github.com/dmlc/tvm.git"
  	GIT_TAG         		"master"
  	PREFIX          		${TVM_SOURCES_DIR}
    DEPENDS             ${LLVM_PROJECT}
    CMAKE_ARGS          -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    CMAKE_ARGS          ${TVM_BUILD_OPTIONS} 
)

add_library(shared_tvm SHARED IMPORTED GLOBAL)
set_property(TARGET shared_tvm PROPERTY	IMPORTED_LOCATION ${TVM_LIB})
add_dependencies(shared_tvm ${TVM_PROJECT} ${LLVM_PROJECT})
list(APPEND external_project_dependencies shared_tvm)

set(dummyfile ${CMAKE_CURRENT_BINARY_DIR}/tvm_dummy.c)
file(WRITE ${dummyfile} "const char * dummy = \"${dummyfile}\";")
add_library(tvm STATIC ${dummyfile})
target_link_libraries(tvm ${TVM_LIB})
add_dependencies(tvm ${TVM_PROJECT})

#set(TVM_SHARED_LIB ${TVM_INSTALL_DIR}/libtvm_runtime.so)
#add_custom_command(OUTPUT ${TVM_SHARED_LIB}
#	COMMAND cp ${TVM_LIB} ${TVM_SHARED_LIB}
#	DEPENDS tvm)
#add_custom_target(tvm_shared_lib DEPENDS ${TVM_SHARD_LIB})

if (WITH_C_API)
	install(FILES ${TVM_LIB} DESTINATION lib)
endif()
