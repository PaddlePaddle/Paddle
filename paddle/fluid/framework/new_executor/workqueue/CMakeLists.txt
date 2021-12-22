cc_library(workqueue SRCS workqueue.cc workqueue_utils.cc DEPS enforce)
cc_test(workqueue_test SRCS workqueue_test.cc DEPS workqueue)
