#pragma once

#if !define(WIN32)
#include <dlfcn.h>     // for dladdr
#include <execinfo.h>  // for backtrace
#endif