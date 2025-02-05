paddle_test(layout_transformation_interface_test SRCS
            layout_transformation_interface_test.cc)
if(WITH_ONNXRUNTIME AND WIN32)
  copy_onnx(layout_transformation_interface_test)
endif()
