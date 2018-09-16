#include "paddle/fluid/framework/naive_executor.h"
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor.h"
#include "naive_executor.h"

namespace paddle {
namespace framework {

// These code can be shared with Executor.
static void InitializeVariable(Variable* var, proto::VarType::Type var_type) {
  if (var_type == proto::VarType::LOD_TENSOR) {
    var->GetMutable<LoDTensor>();
  } else if (var_type == proto::VarType::SELECTED_ROWS) {
    var->GetMutable<SelectedRows>();
  } else if (var_type == proto::VarType::FEED_MINIBATCH) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarType::FETCH_LIST) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarType::STEP_SCOPES) {
    var->GetMutable<std::vector<framework::Scope>>();
  } else if (var_type == proto::VarType::LOD_RANK_TABLE) {
    var->GetMutable<LoDRankTable>();
  } else if (var_type == proto::VarType::LOD_TENSOR_ARRAY) {
    var->GetMutable<LoDTensorArray>();
  } else if (var_type == proto::VarType::PLACE_LIST) {
    var->GetMutable<platform::PlaceList>();
  } else if (var_type == proto::VarType::READER) {
    var->GetMutable<ReaderHolder>();
  } else if (var_type == proto::VarType::CHANNEL) {
    var->GetMutable<ChannelHolder>();
  } else if (var_type == proto::VarType::RAW) {
    // GetMutable will be called in operator
  } else {
    PADDLE_THROW(
        "Variable type %d is not in "
        "[LOD_TENSOR, SELECTED_ROWS, FEED_MINIBATCH, FETCH_LIST, "
        "LOD_RANK_TABLE, PLACE_LIST, READER, CHANNEL, RAW]",
        var_type);
  }
}

void NaiveExecutor::Prepare(Scope* parent_scope,
                            const ProgramDesc& program_desc, int block_id) {
  if (!parent_scope) {
    scope_.reset(new framework::Scope);
  }
  CreateVariables(program_desc, scope_.get(), block_id);
  CreateOps(program_desc, block_id);
}

void NaiveExecutor::Run() {
  for (auto& op : ops_) {
    op->Run(*scope_, place_);
  }
}

void NaiveExecutor::CreateVariables(const ProgramDesc& desc, Scope* scope,
                                    int block_id) {
  PADDLE_ENFORCE(scope);
  auto& global_block = desc.Block(block_id);

  const Scope* ancestor_scope = scope;
  while (ancestor_scope->parent()) {
    ancestor_scope = ancestor_scope->parent();
  }

  if (ancestor_scope != scope) {
    for (auto& var : global_block.AllVars()) {
      if (var->Name() == framework::kEmptyVarName) {
        continue;
      }

      // Create persistable vars in ancestor scope.
      if (var->Persistable()) {
        auto* ptr = const_cast<Scope*>(ancestor_scope)->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
        VLOG(3) << "Create Variable " << var->Name()
                << " global, which pointer is " << ptr;
      } else {  // Create temporary variables in local scope.
        auto* ptr = scope->Var(var->Name());
        InitializeVariable(ptr, var->GetType());
        VLOG(3) << "Create Variable " << var->Name()
                << " locally, which pointer is " << ptr;
      }
    }
  } else {
    for (auto& var : global_block.AllVars()) {
      auto* ptr = scope->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      VLOG(3) << "Create variable " << var->Name() << ", which pointer is "
              << ptr;
    }
  }
}

void NaiveExecutor::CreateOps(const ProgramDesc& desc, int block_id) {
  for (const auto& op_desc : desc.Block(block_id).AllOps()) {
    ops_.emplace_back(OpRegistry::CreateOp(*op_desc));
  }
}

LoDTensor *NaiveExecutor::FindTensor(const std::string &name) {
  PADDLE_ENFORCE(scope_.get(), "Need to init scope first");
  auto* var = scope_->FindVar(name);
  PADDLE_ENFORCE(var, "No variable [%s] in the scope");
  auto* tensor = const_cast<LoDTensor*>(&var->Get<LoDTensor>());
  return tensor;
}

}  // namespace framework
}  // namespace paddle
