#pragma once

#include <string>
#include <stdexcept>

#if !defined(_WIN32)
#include <dlfcn.h>     // for dladdr
#include <execinfo.h>  // for backtrace
#else
#include <Shlwapi.h>
#include <Windows.h>
namespace {

static void* dlsym(void *handle, const char* symbol_name) {
	FARPROC found_symbol;
    found_symbol = GetProcAddress((HMODULE)handle, symbol_name);

    if (found_symbol == NULL) {
    	throw std::runtime_error(std::string(symbol_name) + " not found.");
    }
    return (void*)found_symbol;
}
} // namespace anoymous

#endif