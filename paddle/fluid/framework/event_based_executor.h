#pragma once

#include <string>
#include <memory>
#include <thread>
#include "paddle/fluid/framework/runtime_graph.h"

namespace paddle {
namespace framework {

class ProgramDesc;
class EventBasedWorker;

class EventBasedExecutor {
 public:
	EventBasedExecutor() = default;
	~EventBasedExecutor();

	void Compile(const ProgramDesc& program_desc, const std::string& grain);
	void Run();
 
 private:
	void CompileCoarseGrainGraph(const ProgramDesc& program_desc);
	void CompileFineGrainGraph(const ProgramDesc& program_desc);
	std::unique_ptr<RuntimeGraph> runtime_graph_;
};

}
}
