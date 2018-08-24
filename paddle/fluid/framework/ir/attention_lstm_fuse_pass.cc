#include "paddle/fluid/framework/ir/attention_lstm_fuse_pass.h"
#include "paddle/fluid/framework/ir/graph_pattern_detecter.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/inference/api/helper.h"

namespace paddle {
namespace framework {
namespace ir {

template <typename T>
struct WhilePatternInfo {
  // inputs related.
  T* sequence_expand_op;              //
  T* sequence_expand_out;             // sequence_expand_0.tmp_0
  T* lod_reset_op;                    //
  T* lod_reset_out;                   // lod_reset_0.tmp_0
  T* lod_rank_table_op;               //
  T* lod_rank_table_out;              // lod_rank_table_0
  T* lod_tensor_to_array_op;          //
  T* lod_tensor_to_array_out;         // dynamic_rnn_iput_array_0
  T* fill_constant_op;                //
  T* fill_constant_out;               // fill_constant_1.tmp_0
  T* less_than_op;                    //
  T* less_than_out;                   // dynamic_rnn_0.tmp_0
  T* max_sequence_len_op;             //
  T* max_sequence_len_out;            // dynamic_rnn_max_seq_len_0
  T* fill_constant_op1;               //
  T* fill_constant_out1;              // fill_constant_0.tmp_0
  T* write_to_array_op;               //
  T* write_to_array_out;              // dynamic_rnn_mem_array_1
  T* reorder_lod_tensor_by_rank_op;   //
  T* reorder_lod_tensor_by_rank_out;  // dynamic_rnn_mem_init_reordered_0
  T* write_to_array_op1;              //
  T* write_to_array_out1;             // dynamic_rnn_mem_array_0
  T* cell_init;                       // cell_init
  T* hidden_init;                     // hidden_init
  // outputs related
  T* while_out;  // dynamic_rnn_0_output_array-elementwise_mul_3.tmp_0_0
  T* array_to_lod_tensor_op;   //
  T* array_to_lod_tensor_out;  // array_to_lod_tensor_0.tmp_0
};

void BuildWhileExternalPattern(PDPattern* pattern,
                               WhilePatternInfo<Node>* nodes_info,
                               WhilePatternInfo<PDNode*>* pdnodes_info);
void BuildWhileBlockPattern(PDPattern* pattern,
                            WhilePatternInfo<Node>* nodes_info,
                            WhilePatternInfo<PDNode*>* pdnodes_info);

void FindWhileOp(Graph* graph) {
  GraphPatternDetecter gpd;
  gpd.mutable_pattern()->NewNode(
      [](Node* n) {
        return n->IsOp() && n->Op()->Type() == "while" &&
               n->inputs.size() == 27UL && n->outputs.size() == 6UL;
      },
      "while");

  auto handle = [&](const GraphPatternDetecter::subgraph_t& subgraph,
                    Graph* g) {
    auto* while_pat_node = gpd.pattern().RetriveNode("while");
    auto* while_node = subgraph.at(while_pat_node);
    LOG(INFO) << "find while";
    LOG(INFO) << "inputs " << while_node->inputs.size();
    LOG(INFO) << "outputs " << while_node->outputs.size();
  };
  gpd(graph, handle);
}

void BuildFusePattern(PDPattern* pattern) {}

// Prepare parameters
void PrepareLSTMWeight(const LoDTensor& W_forget, const LoDTensor& W_input,
                       const LoDTensor& W_output, const LoDTensor& W_cell,
                       LoDTensor* out);

void PrepareLSTMBias(const LoDTensor& B_forget, const LoDTensor& B_input,
                     const LoDTensor& B_output, const LoDTensor& B_cell,
                     LoDTensor* out);

void AttentionLSTMFusePass::RegisterParamOperations() const {
  ToCreate("hello");
}

void AttentionLSTMFusePass::Operate(Graph* graph, Scope* scope) const {
  FindWhileOp(graph);
}

bool Contains(const std::vector<Node*>& nodes, const std::string& op_type,
              int count = 1) {
  for (auto* x : nodes) {
    if (x->IsOp() && x->Op() && x->Op()->Type() == op_type) --count;
  }
  return count <= 0;
}

bool ContainsWithFrequency(const std::vector<Node*>& nodes,
                           const std::vector<std::string> op_types) {
  if (op_types.empty()) return true;
  std::map<std::string, int> counts;
  for (const auto& op_type : op_types) {
    ++counts[op_type];
  }
  for (const auto& op_type : op_types) {
    if (!Contains(nodes, op_type, counts.at(op_type))) return false;
  }

  return true;
}

bool VarInLinksToOp(Node* x, const std::vector<std::string>& op_types) {
  return ContainsWithFrequency(x->inputs, op_types);
}

bool VarOutLinksToOp(Node* x, const std::vector<std::string>& op_types) {
  return ContainsWithFrequency(x->outputs, op_types);
}

namespace {

PDNode* CreateOpPNode(PDPattern* pattern, const std::string& op_type) {
  return pattern->NewNode([op_type](Node* x) -> bool {
    return x && x->IsOp() && x->Op()->Type() == op_type;
  });
}

PDNode* CreateVarPDNode(PDPattern* pattern,
                        const std::vector<std::string>& in_op_types,
                        const std::vector<std::string>& out_op_types) {
  return pattern->NewNode([in_op_types, out_op_types](Node* x) -> bool {
    return x && x->IsVar() && VarInLinksToOp(x, in_op_types) &&
           VarOutLinksToOp(x, out_op_types);
  });
}

#define LINKS(a, b) pattern->AddEdge(a, b);

// simulate the pythoin DynamicRNN
PDNode* LoDTensorToArray(PDPattern* pattern, PDNode* x, PDNode* table) {
  auto* op = CreateOpPNode(pattern, "lod_tensor_to_array");
  auto* array = CreateVarPDNode(pattern, {"lod_tensor_to_array"}, {});
  // input
  LINKS(x, op);
  LINKS(table, op);
  // output
  LINKS(op, array);
  return array;
}

PDNode* ArrayToLoDTensor(PDPattern* pattern, PDNode* x, PDNode* table) {
  auto* op = CreateOpPNode(pattern, "array_to_lod_tensor");
  auto* tmp = CreateVarPDNode(pattern, {"array_to_lod_tensor"}, {});
  // inputs
  LINKS(x, op);
  LINKS(table, op);
  // outputs
  LINKS(op, tmp);
  return tmp;
}

PDNode* Increment(PDPattern* pattern, PDNode* x) {
  auto* op = CreateOpPNode(pattern, "increment");
  auto* out = CreateVarPDNode(pattern, {"increment"}, {});
  // inputs
  LINKS(x, op);
  // outputs
  LINKS(op, out);
  return out;
}

PDNode* ArrayWrite(PDPattern* pattern, PDNode* x, PDNode* i,
                   PDNode* array = nullptr) {
  if (!array) {
    array = CreateVarPDNode(pattern, {}, {});
  }
  auto* op = CreateOpPNode(pattern, "write_to_array");
  LINKS(x, op);
  LINKS(i, op);
  LINKS(op, array);
  return array;
}

PDNode* CreateArray(PDPattern* pattern) {
  return CreateVarPDNode(pattern, {}, {});
}

PDNode* LessThan(PDPattern* pattern, PDNode* x, PDNode* y) {
  auto* op = CreateOpPNode(pattern, "less_than");
  auto* out = CreateVarPDNode(pattern, {"less_than"}, {});
  LINKS(x, op);
  LINKS(y, op);
  LINKS(op, out);
  return out;
}

PDNode* Equal(PDPattern* pattern, PDNode* x, PDNode* y) {
  auto* op = CreateOpPNode(pattern, "equal");
  auto* out = CreateO
}



}  // namespace

// void BuildWhileExternalPattern(PDPattern* pattern,
//                                WhilePatternInfo<Node>* nodes_info,
//                                WhilePatternInfo<PDNode*>* pdnodes_info) {
// // Create operators
// #define CREATE_OP_PNODE(key, op_type)   \
//   pdnodes_info->key = pattern->NewNode( \
//       [](Node* x) { return x && x->IsOp() && x->Op()->Type() == #op_type });
//   CREATE_OP_PNODE(sequence_expand_op, sequence_expand);
//   CREATE_OP_PNODE(lod_reset_op, lod_reset);
//   CREATE_OP_PNODE(lod_rank_table_op, lod_rank_table);
//   CREATE_OP_PNODE(lod_tensor_to_array_op, lod_tensor_to_array);
//   CREATE_OP_PNODE(fill_constant_op, fill_constant);
//   CREATE_OP_PNODE(less_than_op, less_than);
//   CREATE_OP_PNODE(max_sequence_len_op, max_sequence_len);
//   CREATE_OP_PNODE(fill_constant_op1, fill_constant);
//   CREATE_OP_PNODE(write_to_array_op, write_to_array);
//   CREATE_OP_PNODE(reorder_lod_tensor_by_rank_op, reorder_lod_tensor_by_rank);
//   CREATE_OP_PNODE(write_to_array_op1, write_to_array);
//   CREATE_OP_PNODE(array_to_lod_tensor_op, array_to_lod_tensor);
// #undef CREATE_OP_PNODE

// // Create variables
// #define CREATE_VAR_PNODE(key, in_op, out_op)                  \
//   pdnodes_info->key = pattern->NewNode([](Node* x) {          \
//     return x && x->IsVar() &&                                 \
//            VarInLinksToOp(x, inference::split(in_op, ',')) && \
//            VarOutLinksToOp(x, inference::split(out_op, ',')); \
//   });
//   CREATE_VAR_PNODE(sequence_expand_out, "sequence_expand", "lod_reset");
//   CREATE_VAR_PNODE(lod_reset_out, "lod_reset",
//                    "lod_tensor_to_array,lod_rank_table");
//   CREATE_VAR_PNODE(lod_rank_table_out, "lod_rank_table",
//                    "lod_tensor_to_array,array_to_lod_tensor,reorder_lod_tensor_"
//                    "by_rank,max_sequence_len");
//   CREATE_VAR_PNODE(fill_constant_out, "fill_constant", "less_than,while");
//   CREATE_VAR_PNODE(max_sequence_len_out, "max_sequence_len",
//   "less_than,while");
//   CREATE_VAR_PNODE(less_than_out, "less_than", "while");
//   CREATE_VAR_PNODE(fill_constant_out1, "fill_constant",
//                    "write_to_array,write_to_array");
//   CREATE_VAR_PNODE(while_out, "while", "array_to_lod_tensor");
//   CREATE_VAR_PNODE(array_to_lod_tensor_out, "array_to_lod_tensor",
//                    "sequence_expand,sequence_expand");
//   CREATE_VAR_PNODE(lod_tensor_to_array_out, "lod_tensor_to_array", "while");
// // TODO(Superjomn) add two write_to_array_outs

// #undef CREATE_VAR_PNODE
// }

void BuildWhileBlockPattern(PDPattern* pattern,
                            WhilePatternInfo<Node>* nodes_info,
                            WhilePatternInfo<PDNode*>* pdnodes_info);

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(attention_lstm_fuse_pass,
              paddle::framework::ir::AttentionLSTMFusePass);
