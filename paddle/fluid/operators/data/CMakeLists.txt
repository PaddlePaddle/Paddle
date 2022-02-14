include(operators)
if(WITH_UNITY_BUILD)
    # Load Unity Build rules for operators in paddle/fluid/operators/data/
    include(unity_build_rule.cmake)
endif()

cc_library(pipeline SRCS pipeline.cc DEPS parallel_executor simple_threadpool scope)
op_library(dataloader_op SRCS dataloader_op.cc dataloader_op.cu.cc DEPS pipeline ${OP_HEADER_DEPS})

op_library(data_reader_op SRCS data_reader_op.cc DEPS ${OP_HEADER_DEPS})

cc_library(map_runner SRCS map_runner.cc DEPS parallel_executor simple_threadpool scope)
op_library(map_op SRCS map_op.cc map_op.cu.cc DEPS map_runner ${OP_HEADER_DEPS})

cc_library(random_roi_generator SRCS random_roi_generator.cc DEPS ${OP_HEADER_DEPS})
cc_library(nvjpeg_decoder SRCS nvjpeg_decoder.cc DEPS random_roi_generator ${OP_HEADER_DEPS} ${OpenCV_LIBS})
op_library(batch_decode_random_crop_op SRCS batch_decode_random_crop_op.cc batch_decode_random_crop_op.cu DEPS nvjpeg_decoder ${OP_HEADER_DEPS})
op_library(batch_decode_op SRCS batch_decode_op.cc batch_decode_op.cu DEPS nvjpeg_decoder ${OP_HEADER_DEPS})

op_library(random_crop_and_resize_op SRCS random_crop_and_resize_op.cc random_crop_and_resize_op.cu DEPS ${OP_HEADER_DEPS})
op_library(batch_resize_op SRCS batch_resize_op.cc batch_resize_op.cu DEPS ${OP_HEADER_DEPS})

op_library(file_label_loader_op SRCS file_label_loader_op.cc DEPS ${OP_HEADER_DEPS})

# register_operators()

# TODO: add test here
# cc_test(xxx SRCS xxx DEPS xxx