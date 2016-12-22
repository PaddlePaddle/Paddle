licenses(["notice"])

cc_library(
    visibility=["//visibility:public"],
    name="glog",
    includes=[
        ".",
        "src",
    ],
    copts=[
        "-D_START_GOOGLE_NAMESPACE_='namespace google {'",
        "-D_END_GOOGLE_NAMESPACE_='}'",
        "-DGOOGLE_NAMESPACE='google'",
        "-DGOOGLE_GLOG_DLL_DECL=''",
        "-DHAVE_DLADDR",
        "-DHAVE_SNPRINTF",
        "-DHAVE_DLFCN_H",
        "-DHAVE_FCNTL",
        "-DHAVE_GLOB_H",
        "-DHAVE_INTTYPES_H",
        "-DHAVE_LIBPTHREAD",
        "-DHAVE_SYS_SYSCALL_H",
        "-DHAVE_MEMORY_H",
        "-DHAVE_NAMESPACES",
        "-DHAVE_PREAD",
        "-DHAVE_PTHREAD",
        "-DHAVE_PWD_H",
        "-DHAVE_PWRITE",
        "-DHAVE_RWLOCK",
        "-DHAVE_SIGACTION",
        "-DHAVE_SIGALTSTACK",
        "-DHAVE_STDINT_H",
        "-DHAVE_STRING_H",
        "-DHAVE_SYS_TIME_H",
        "-DHAVE_SYS_TYPES_H",
        "-DHAVE_SYS_UCONTEXT_H",
        "-DHAVE_SYS_UTSNAME_H",
        "-DHAVE_UNISTD_H",
        "-DHAVE_USING_OPERATOR",
        "-DHAVE_HAVE___ATTRIBUTE___",
        "-DHAVE_HAVE___BUILTIN_EXPECT",
        #"-DNO_FRAME_POINTER",
        "-D_GNU_SOURCE",
        #"-fno-sanitize=thread",
        #"-fno-sanitize=address",
        "-Iexternal/glog/src",
    ],
    srcs=[
        "src/demangle.cc",
        "src/logging.cc",
        "src/raw_logging.cc",
        "src/signalhandler.cc",
        "src/symbolize.cc",
        "src/utilities.cc",
        "src/vlog_is_on.cc",
        ":config_h",
        ":logging_h",
        ":raw_logging_h",
        ":stl_logging_h",
        ":vlog_is_on_h",
    ],
    hdrs=[
        "src/demangle.h",
        "src/mock-log.h",
        "src/stacktrace.h",
        "src/symbolize.h",
        "src/utilities.h",
        "src/base/commandlineflags.h",
        "src/base/googleinit.h",
        "src/base/mutex.h",
        "src/glog/log_severity.h",
    ])

genrule(
    name="config_h",
    srcs=["src/config.h.cmake.in"],
    outs=["config.h"],
    cmd="awk '{ gsub(/^#cmakedefine/, \"//cmakedefine\"); print; }' $(<) > $(@)",
)

genrule(
    name="logging_h",
    srcs=["src/glog/logging.h.in"],
    outs=["glog/logging.h"],
    cmd="$(location :gen_sh) < $(<) > $(@)",
    tools=[":gen_sh"])

genrule(
    name="raw_logging_h",
    srcs=["src/glog/raw_logging.h.in"],
    outs=["glog/raw_logging.h"],
    cmd="$(location :gen_sh) < $(<) > $(@)",
    tools=[":gen_sh"])

genrule(
    name="stl_logging_h",
    srcs=["src/glog/stl_logging.h.in"],
    outs=["glog/stl_logging.h"],
    cmd="$(location :gen_sh) < $(<) > $(@)",
    tools=[":gen_sh"])

genrule(
    name="vlog_is_on_h",
    srcs=["src/glog/vlog_is_on.h.in"],
    outs=["glog/vlog_is_on.h"],
    cmd="$(location :gen_sh) < $(<) > $(@)",
    tools=[":gen_sh"])

genrule(
    name="gen_sh",
    outs=["gen.sh"],
    cmd="""
cat > $@ <<"EOF"
#! /bin/sh
sed -e 's/@ac_cv_have_unistd_h@/1/g' \
    -e 's/@ac_cv_have_stdint_h@/1/g' \
    -e 's/@ac_cv_have_systypes_h@/1/g' \
    -e 's/@ac_cv_have_libgflags_h@/1/g' \
    -e 's/@ac_cv_have_uint16_t@/1/g' \
    -e 's/@ac_cv_have___builtin_expect@/1/g' \
    -e 's/@ac_cv_have_.*@/0/g' \
    -e 's/@ac_google_start_namespace@/namespace google {/g' \
    -e 's/@ac_google_end_namespace@/}/g' \
    -e 's/@ac_google_namespace@/google/g' \
    -e 's/@ac_cv___attribute___noinline@/__attribute__((noinline))/g' \
    -e 's/@ac_cv___attribute___noreturn@/__attribute__((noreturn))/g' \
    -e 's/@ac_cv___attribute___printf_4_5@/__attribute__((__format__ (__printf__, 4, 5)))/g'
EOF""")
