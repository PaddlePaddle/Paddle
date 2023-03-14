if(NOT WITH_MLU)
  return()
endif()

if(NOT ENV{NEUWARE_HOME})
  set(NEUWARE_HOME "/usr/local/neuware")
else()
  set(NEUWARE_HOME $ENV{NEUWARE_HOME})
endif()
message(STATUS "NEUWARE_HOME: " ${NEUWARE_HOME})

set(NEUWARE_INCLUDE_DIR ${NEUWARE_HOME}/include)
set(NEUWARE_LIB_DIR ${NEUWARE_HOME}/lib64)

include_directories(${NEUWARE_INCLUDE_DIR})

set(CNNL_LIB ${NEUWARE_LIB_DIR}/libcnnl.so)
set(MLUOP_LIB ${NEUWARE_LIB_DIR}/libmluops.so)
set(CNRT_LIB ${NEUWARE_LIB_DIR}/libcnrt.so)
set(CNDRV_LIB ${NEUWARE_LIB_DIR}/libcndrv.so)
set(CNPAPI_LIB ${NEUWARE_LIB_DIR}/libcnpapi.so)

generate_dummy_static_lib(LIB_NAME "neuware_lib" GENERATOR "neuware.cmake")
set(NEUWARE_LIB_DEPS ${CNNL_LIB} ${MLUOP_LIB} ${CNRT_LIB} ${CNDRV_LIB}
                     ${CNPAPI_LIB})

if(WITH_CNCL)
  message(STATUS "Compile with CNCL!")
  add_definitions(-DPADDLE_WITH_CNCL)
  set(CNCL_LIB ${NEUWARE_LIB_DIR}/libcncl.so)
  list(APPEND NEUWARE_LIB_DEPS ${CNCL_LIB})
endif()

target_link_libraries(neuware_lib ${NEUWARE_LIB_DEPS})
