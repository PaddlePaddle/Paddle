#include <string>
#include "paddle/fluid/framework/event_based_executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/runtime_graph.h"

namespace paddle {
namespace framework {

EventBasedExecutor::~EventBasedExecutor() {
	std::cout << "In EventBased Deconstructor" << std::endl;
}

void EventBasedExecutor::Compile(const ProgramDesc& program, const std::string& grain) {
	std::cout << "In Event Based Executor Compile" << std::endl;
	if (grain == "coarse") {
		CompileCoarseGrainGraph(program);
	} else {
		CompileFineGrainGraph(program);
	}
}

void EventBasedExecutor::CompileCoarseGrainGraph(const ProgramDesc& program) {
	std::cout << "Compile Coarse Grain Graph" << std::endl;
	runtime_graph_.reset(new RuntimeGraph(program));
	runtime_graph_->PrintGraph();
}

void EventBasedExecutor::CompileFineGrainGraph(const ProgramDesc& program) {
	std::cout << "Compile Fine Grain Graph" << std::endl;
}

void EventBasedExecutor::Run() {
	std::cout << "In Event Based Executor Run" << std::endl;
}

}
}
