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

INCLUDE_DIRECTORIES(${NEUWARE_INCLUDE_DIR})

set(CNNL_LIB ${NEUWARE_LIB_DIR}/libcnnl.so)
set(CNRT_LIB ${NEUWARE_LIB_DIR}/libcnrt.so)
set(CNDRV_LIB ${NEUWARE_LIB_DIR}/libcndrv.so)

generate_dummy_static_lib(LIB_NAME "neuware_lib" GENERATOR "neuware.cmake")
TARGET_LINK_LIBRARIES(neuware_lib ${CNNL_LIB} ${CNRT_LIB} ${CNDRV_LIB})
