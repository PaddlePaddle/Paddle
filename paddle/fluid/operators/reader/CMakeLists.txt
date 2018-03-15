cc_library(reader_op_registry SRCS reader_op_registry.cc DEPS operator op_registry reader)
set(LOCAL_READER_LIBS)

function(reader_library TARGET_NAME)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    set(options "")
    set(common_deps reader_op_registry)
    cmake_parse_arguments(reader_library "${options}" "${oneValueArgs}"
            "${multiValueArgs}" ${ARGN})
    op_library(${TARGET_NAME} SRCS ${reader_library_SRCS} DEPS ${common_deps} ${reader_library_DEPS})
    set(LOCAL_READER_LIBS
            ${TARGET_NAME}
            ${LOCAL_READER_LIBS}
        PARENT_SCOPE)
endfunction()

reader_library(create_random_data_generator_op SRCS create_random_data_generator_op.cc)
reader_library(create_shuffle_reader_op SRCS create_shuffle_reader_op.cc)
reader_library(create_batch_reader_op SRCS create_batch_reader_op.cc)
reader_library(create_recordio_file_reader_op SRCS create_recordio_file_reader_op.cc)
reader_library(create_double_buffer_reader_op SRCS create_double_buffer_reader_op.cc)
# Export local libraries to parent
set(READER_LIBRARY ${LOCAL_READER_LIBS} PARENT_SCOPE)
