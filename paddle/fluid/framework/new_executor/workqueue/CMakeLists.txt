cc_library(workqueue SRCS workqueue.cc workqueue_utils.cc events_waiter.cc DEPS enforce glog)
cc_test(workqueue_test SRCS workqueue_test.cc DEPS workqueue)
