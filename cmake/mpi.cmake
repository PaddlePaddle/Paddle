if(NOT WITH_DISTRIBUTE OR NOT WITH_MPI)
  return()
endif()

find_package(MPI)

if(NOT MPI_CXX_FOUND)
  set(WITH_MPI
      OFF
      CACHE STRING "Disable MPI" FORCE)
  message(WARNING "Not found MPI support in current system")
  return()
endif()

message(STATUS "MPI compile flags: " ${MPI_CXX_COMPILE_FLAGS})
message(STATUS "MPI include path: " ${MPI_CXX_INCLUDE_PATH})
message(STATUS "MPI LINK flags path: " ${MPI_CXX_LINK_FLAGS})
message(STATUS "MPI libraries: " ${MPI_CXX_LIBRARIES})
include_directories(SYSTEM ${MPI_CXX_INCLUDE_PATH})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_CXX_LINK_FLAGS}")
add_definitions("-DPADDLE_WITH_MPI")
find_program(
  OMPI_INFO
  NAMES ompi_info
  HINTS ${MPI_CXX_LIBRARIES}/../bin)

if(OMPI_INFO)
  execute_process(COMMAND ${OMPI_INFO} OUTPUT_VARIABLE output_)
  if(output_ MATCHES "smcuda")
    #NOTE some mpi lib support mpi cuda aware.
    add_definitions("-DPADDLE_WITH_MPI_AWARE")
  endif()
endif()
