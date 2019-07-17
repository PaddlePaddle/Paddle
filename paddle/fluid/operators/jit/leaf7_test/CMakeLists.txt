cmake_minimum_required(VERSION 2.6)
project(leaf7_test)
ENABLE_LANGUAGE(ASM)

cc_test(leaf7_test SRCS leaf7_test.c read_cpuid.s)