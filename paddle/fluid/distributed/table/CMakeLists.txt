set_property(GLOBAL PROPERTY TABLE_DEPS string_helper)

get_property(TABLE_DEPS GLOBAL PROPERTY TABLE_DEPS)

set_source_files_properties(common_dense_table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(common_sparse_table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(sparse_geo_table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(barrier_table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

cc_library(common_table SRCS common_sparse_table.cc common_dense_table.cc sparse_geo_table.cc barrier_table.cc DEPS ${TABLE_DEPS} device_context string_helper simple_threadpool xxhash generator)

set_source_files_properties(tensor_accessor.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
set_source_files_properties(tensor_table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})
cc_library(tensor_accessor SRCS tensor_accessor.cc DEPS ${TABLE_DEPS} eigen3 ps_framework_proto device_context)
cc_library(tensor_table SRCS tensor_table.cc DEPS ps_framework_proto proto_desc enforce executor tensor device_context simple_threadpool gflags glog )

set_source_files_properties(table.cc PROPERTIES COMPILE_FLAGS ${DISTRIBUTE_COMPILE_FLAGS})

cc_library(table SRCS table.cc DEPS common_table tensor_table tensor_accessor ps_framework_proto string_helper device_context gflags glog boost)
