#pragma once

namespace paddle {

namespace inference {

int RegisterPyFunc(const std::string& name, void* pfn);
}
}