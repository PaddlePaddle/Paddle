# user should download rdma first from subversion repository

# execute following instruction to download svn mannally
# svn co https://svn.baidu.com/sys/ip/trunk/rdma/sockrdmav1 rdma/
# svn co https://svn.baidu.com/sys/ip/trunk/rdma/thirdparty rdma/
# we use static output in svn repositories to avoid implict bugs from not standard runtime env.

set(RDMA_ROOT $ENV{RDMA_ROOT} CACHE PATH "Folder contains RDMA sock library and thirdparty library")


#check and set headers
find_path(RDMA_INC_SXISOCK sxi_sock.h PATHS ${RDMA_ROOT}/sockrdmav1/output/include)
find_path(RDMA_INC_XIO libxio.h PATHS ${RDMA_ROOT}/thirdparty/output/accelio)
find_path(RDMA_INC_EVENT event2 PATHS ${RDMA_ROOT}/thirdparty/output/libevent)
find_path(RDMA_INC_NUMA numa.h PATHS ${RDMA_ROOT}/thirdparty/output/libnuma)

#check and set libs
find_library(RDMA_LIB_SXISOCK NAMES sxisock PATHS ${RDMA_ROOT}/sockrdmav1/output)
find_library(RDMA_LIB_XIO NAMES xio PATHS ${RDMA_ROOT}/thirdparty/output/accelio)
find_library(RDMA_LIB_EVENT NAMES event PATHS ${RDMA_ROOT}/thirdparty/output/libevent)
find_library(RDMA_LIB_EVENT_CORE NAMES event_core PATHS ${RDMA_ROOT}/thirdparty/output/libevent)
find_library(RDMA_LIB_EVENT_EXTRA NAMES event_extra PATHS ${RDMA_ROOT}/thirdparty/output/libevent)
find_library(RDMA_LIB_EVENT_PTHREADS NAMES event_pthreads PATHS ${RDMA_ROOT}/thirdparty/output/libevent)
find_library(RDMA_LIB_NUMA NAMES numa PATHS ${RDMA_ROOT}/thirdparty/output/libnuma)

if(
    RDMA_INC_SXISOCK AND
    RDMA_INC_XIO AND
    RDMA_INC_EVENT AND
    RDMA_INC_NUMA AND
    RDMA_LIB_SXISOCK AND 
    RDMA_LIB_XIO AND
    RDMA_LIB_EVENT AND
    RDMA_LIB_EVENT_CORE AND
    RDMA_LIB_EVENT_EXTRA AND
    RDMA_LIB_EVENT_PTHREADS AND
    RDMA_LIB_NUMA
    )

  set(RDMA_INC_DIR 
    ${RDMA_INC_SXISOCK} 
    ${RDMA_INC_XIO}
    ${RDMA_INC_EVENT}
    ${RDMA_INC_NUMA})
  set(RDMA_LIBS  
    ${RDMA_LIB_SXISOCK} 
    ${RDMA_LIB_XIO} 
    ${RDMA_LIB_EVENT} 
    ${RDMA_LIB_EVENT_CORE} 
    ${RDMA_LIB_EVENT_EXTRA} 
    ${RDMA_LIB_EVENT_PTHREADS} 
    ${RDMA_LIB_NUMA} 
    )
  set(RDMA_LD_FLAGS "-L./librdma -libverbs -lrdmacm -Xlinker -rpath ./librdma")
  return()
endif()

#if this module is not called, RDMA_INC_DIR RDMA_LIBS will be null, so top module always refer this variable


message(FATAL_ERROR, "RDMA libraries are not found, try to set RDMA_ROOT or check all related libraries.")
