cc_library(checkpoint_handler SRCS checkpoint_handler.cc DEPS scope)
cc_library(get_handler SRCS get_handler.cc DEPS scope)
cc_library(prefetch_handler SRCS prefetch_handler.cc DEPS scope)
cc_library(send_handler SRCS send_handler.cc DEPS scope)
cc_library(get_monomer_handler SRCS get_monomer_handler.cc DEPS scope)
cc_library(get_no_barrier_handler SRCS get_no_barrier_handler.cc DEPS scope)
cc_library(signal_handler SRCS signal_handler.cc DEPS scope)
cc_library(handlers DEPS checkpoint_handler get_handler prefetch_handler
  send_handler get_monomer_handler signal_handler get_no_barrier_handler)
