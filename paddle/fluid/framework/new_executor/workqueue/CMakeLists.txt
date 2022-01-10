cc_library(workqueue_utils SRCS workqueue_utils.cc events_waiter.cc DEPS enforce glog)
cc_library(workqueue SRCS workqueue.cc DEPS workqueue_utils enforce glog)
cc_test(workqueue_test SRCS workqueue_test.cc DEPS workqueue)
