#include "paddle/fluid/framework/runtime_graph.h"
#include "paddle/fluid/framework/program_desc.h"

namespace paddle {
namespace framework {

namespace {
typedef std::unordered_map<std::string, std::vector<int64_t>> VarList;

void FilterAndAddOutputVars(const BlockDesc &block,
                            const VarList &prev_input_vars,
                            VarList *cur_output_vars) {
  const auto &ops_in_block = block.AllOps();
  for (const OpDesc *op : ops_in_block) {
    const auto &output_names = op->OutputArgumentNames();
    for (const auto &output_name : output_names) {
      if (prev_input_vars.find(output_name) != prev_input_vars.end()) {
        for (int64_t consumer_id : prev_input_vars.at(output_name)) {
          if (cur_output_vars->find(output_name) == cur_output_vars->end()) {
            cur_output_vars->emplace(output_name, std::vector<int64_t>());
          }
          cur_output_vars->at(output_name).emplace_back(consumer_id);
        }
      }
    }
  }
}

void CreateVarNodesAndAddDeps(const VarList &cur_output_vars,
                              int64_t producer_id,
                              RuntimeGraph *runtime_graph) {
  TaskNode *producer = runtime_graph->GetTaskNode(producer_id);
  for (const auto &output_var : cur_output_vars) {
    const auto &var_name = output_var.first;
    if (!runtime_graph->HasVarNode(var_name)) {
      VarDesc *var = producer->FindVar(var_name);
      runtime_graph->CreateAndAddVarNode(*var);
    }
    InterVarNode *var_node = runtime_graph->FindVarNode(var_name);
    for (int64_t consumer_id : cur_output_vars.at(var_name)) {
      TaskNode *consumer = runtime_graph->GetTaskNode(consumer_id);
      var_node->AddConsumedTask(consumer_id);
      var_node->SetProducedTask(producer_id);
      producer->AddProducedVarNode(var_name);
      consumer->AddConsumedVarNode(var_name);
    }
  }
}

void FilterAndAddInputVars(const BlockDesc &block, VarList *prev_input_vars) {
  const auto &ops_in_block = block.AllOps();
  for (const OpDesc *op : ops_in_block) {
    const auto &var_names = op->InputArgumentNames();
    for (const auto &name : var_names) {
      if (prev_input_vars->find(name) == prev_input_vars->end()) {
        prev_input_vars->emplace(name, std::vector<int64_t>());
      }
      prev_input_vars->at(name).emplace_back(block.ID());
    }
  }
}
}

InterVarNode::InterVarNode(const VarDesc &var) : name_(var.Name()), var_(&var) {
  task_produce_this_var_ = -1;
}

void InterVarNode::AddConsumedTask(int64_t task_id) {
  tasks_consume_this_var_.insert(task_id);
}

void InterVarNode::SetProducedTask(int64_t task_id) {
  task_produce_this_var_ = task_id;
}

TaskNode::TaskNode(const BlockDesc &block)
    : task_id_(block.ID()), block_(&block) {}

void TaskNode::AddConsumedVarNode(const std::string &var_node_name) {
  consumed_var_node_names_.insert(var_node_name);
}

void TaskNode::AddProducedVarNode(const std::string &var_node_name) {
  produced_var_node_names_.insert(var_node_name);
}

bool TaskNode::IsSrcTask() const { return consumed_var_node_names_.empty(); }

VarDesc *TaskNode::FindVar(const std::string &name) const {
  return block_->FindVar(name);
}

void TaskNode::PrintTaskNode() const {
  std::cout << "consumed variables"
            << ": ";
  for (const auto &name : consumed_var_node_names_) {
    std::cout << name << " ";
  }
  std::cout << std::endl;
  std::cout << "produced variables"
            << ": ";
  for (const auto &name : produced_var_node_names_) {
    std::cout << name << " ";
  }
  std::cout << std::endl;
}

RuntimeGraph::RuntimeGraph(const ProgramDesc &program) {
  int64_t block_size = program.Size();
  task_nodes_.resize(block_size);
  VarList prev_input_vars;
  for (int64_t i = block_size - 1; i >= 0; --i) {
    const auto &block = program.Block(i);
    task_nodes_[i].reset(new TaskNode(block));
    VarList cur_output_vars;
    FilterAndAddOutputVars(block, prev_input_vars, &cur_output_vars);
    CreateVarNodesAndAddDeps(cur_output_vars, block.ID(), this);
    FilterAndAddInputVars(block, &prev_input_vars);
  }
}

bool RuntimeGraph::HasVarNode(const std::string &name) const {
  return var_nodes_.find(name) != var_nodes_.end();
}

InterVarNode *RuntimeGraph::FindVarNode(const std::string &name) const {
  CHECK(var_nodes_.find(name) != var_nodes_.end());
  return var_nodes_.at(name).get();
}

TaskNode *RuntimeGraph::GetTaskNode(int64_t id) const {
  CHECK_LT(id, (int)task_nodes_.size());
  return task_nodes_[id].get();
}

InterVarNode *RuntimeGraph::CreateAndAddVarNode(const VarDesc &var) {
  InterVarNode *var_node = new InterVarNode(var);
  var_nodes_.emplace(var.Name(), var_node);
  return var_node;
}

int64_t RuntimeGraph::TaskNodesNum() const { return task_nodes_.size(); }

void RuntimeGraph::PrintGraph() const {
  for (const auto &task : task_nodes_) {
    task->PrintTaskNode();
  }
}
}
}
