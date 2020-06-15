include(ExternalProject)
include_directories(${INSTALL_DIR})

set(URL "https://paddle-ci.cdn.bcebos.com")
set(FILENAME "buddy_allocator_test_data.tar")
set(INSTALL_DIR ${THIRD_PARTY_PATH}/buddy_allocator)


cc_library(memory_block SRCS memory_block.cc memory_block_desc.cc meta_cache.cc DEPS place)

if(${WITH_GPU})
  nv_library(system_allocator SRCS system_allocator.cc DEPS gflags cpu_info gpu_info place)
else(${WITH_GPU})
  cc_library(system_allocator SRCS system_allocator.cc DEPS gflags cpu_info place)
endif(${WITH_GPU})

cc_test(system_allocator_test SRCS system_allocator_test.cc DEPS system_allocator)

cc_library(buddy_allocator SRCS buddy_allocator.cc DEPS memory_block system_allocator glog)

cc_test(buddy_allocator_test SRCS buddy_allocator_test.cc DEPS buddy_allocator)

function(download_and_uncompress INSTALL_DIR URL FILENAME)
  message(STATUS "Download test stuff from ${URL}/${FILENAME}")
  ExternalProject_Add(
      buddy_allocator_data
      PREFIX                ${INSTALL_DIR}
      DOWNLOAD_COMMAND      wget --no-check-certificate -q -O ${INSTALL_DIR}/${FILENAME} ${URL}/${FILENAME} &&
                            ${CMAKE_COMMAND} -E tar xzf ${INSTALL_DIR}/${FILENAME}
      DOWNLOAD_DIR          ${INSTALL_DIR}
      DOWNLOAD_NO_PROGRESS  1
      CONFIGURE_COMMAND     ""
      BUILD_COMMAND         ""
      INSTALL_COMMAND       ""
      UPDATE_COMMAND        ""
  )
endfunction()


if(WITH_TESTING)
  set_tests_properties(buddy_allocator_test PROPERTIES LABELS "RUN_TYPE=EXCLUSIVE")
  add_dependencies(buddy_allocator buddy_allocator_data)
  if (NOT EXISTS ${INSTALL_DIR}/${FILENAME})
      download_and_uncompress(${INSTALL_DIR} ${URL} ${FILENAME})
  endif()
endif()
