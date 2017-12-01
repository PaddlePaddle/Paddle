cc_library(ThreadLocal SRCS ThreadLocal.cc)
cc_library(Stat SRCS Stat.cc DEPS ThreadLocal)
