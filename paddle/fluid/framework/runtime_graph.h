#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace paddle {
namespace framework {

class ProgramDesc;
class OpDesc;
class BlockDesc;
class VarDesc;

class InterVarNode final {
 public:
	InterVarNode() = delete;
	explicit InterVarNode(const VarDesc& var);
	~InterVarNode() = default;
	InterVarNode(const InterVarNode&) = delete;
	InterVarNode(InterVarNode&&) = delete;
	InterVarNode& operator=(const InterVarNode&) = delete;
	InterVarNode& operator=(InterVarNode&&) = delete;

	void AddConsumedTask(int64_t task_id);
	void SetProducedTask(int64_t task_id);

 private:
	std::string name_;
	std::unordered_set<int64_t> tasks_consume_this_var_;
	int64_t task_produce_this_var_;
	const VarDesc* var_;
};

class TaskNode final {
 public:
	TaskNode() = delete;
	explicit TaskNode(const BlockDesc& block);
	~TaskNode() = default;
	TaskNode(const TaskNode&) = delete;
	TaskNode(TaskNode&&) = delete;
	TaskNode& operator=(const TaskNode&) = delete;
	TaskNode& operator=(TaskNode&&) = delete;

	bool IsSrcTask() const;
	void AddConsumedVarNode(const std::string& var_node_name);
	void AddProducedVarNode(const std::string& var_node_name);
	VarDesc* FindVar(const std::string& name) const;
	void PrintTaskNode() const;

 private:
	int64_t task_id_;
	const BlockDesc* block_;
	std::unordered_set<std::string> consumed_var_node_names_;
	std::unordered_set<std::string> produced_var_node_names_;
};

class RuntimeGraph final {
 public:
	RuntimeGraph() = delete;
	explicit RuntimeGraph(const ProgramDesc& program); 
	~RuntimeGraph() = default;
	RuntimeGraph(const RuntimeGraph&) = delete;
	RuntimeGraph(RuntimeGraph&&) = delete;
	RuntimeGraph& operator=(const RuntimeGraph&) = delete;
	RuntimeGraph& operator=(RuntimeGraph&&) = delete;

	TaskNode* GetTaskNode(int64_t id) const;
	InterVarNode* FindVarNode(const std::string& name) const;
	bool HasVarNode(const std::string& name) const;
	InterVarNode* CreateAndAddVarNode(const VarDesc& var);
	int64_t TaskNodesNum() const;
	void PrintGraph() const;

 private:
	std::vector<std::unique_ptr<TaskNode>> task_nodes_;
	std::unordered_map<std::string, std::unique_ptr<InterVarNode>> var_nodes_;
};
}
}
