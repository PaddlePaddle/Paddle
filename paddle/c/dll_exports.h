#pragma once

// Since we only support linux and macos in compile, always use clang or
// gcc 4.8+. DLL_IMPORT/DLL_EXPORT is as simple as below.
#define PADDLE_API __attribute__((visibility("default")))
