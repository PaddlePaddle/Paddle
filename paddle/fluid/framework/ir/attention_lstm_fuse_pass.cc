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

#define NO_ARG_OP(func_name__, op_type__)                   \
  PDNode* func_name__(PDPattern* pattern) {                 \
    auto* out = CreateVarPDNode(pattern, {#op_type__}, {}); \
    return out;                                             \
  }

#define ONE_ARG_OP(func_name__, op_type__)                  \
  PDNode* func_name__(PDPattern* pattern, PDNode* x) {      \
    auto* op = CreateOpPNode(pattern, #op_type__);          \
    auto* out = CreateVarPDNode(pattern, {#op_type__}, {}); \
    LINKS(x, op);                                           \
    LINKS(op, out);                                         \
    return out;                                             \
  }

#define TWO_ARG_OP(func_name__, op_type__)                        \
  PDNode* func_name__(PDPattern* pattern, PDNode* x, PDNode* y) { \
    auto* op = CreateOpPNode(pattern, #op_type__);                \
    auto* out = CreateVarPDNode(pattern, {#op_type__}, {});       \
    LINKS(x, op);                                                 \
    LINKS(y, op);                                                 \
    LINKS(op, out);                                               \
    return out;                                                   \
  }
#define THREE_ARG_OP(func_name__, op_type__)                                 \
  PDNode* func_name__(PDPattern* pattern, PDNode* x, PDNode* y, PDNode* z) { \
    auto* op = CreateOpPNode(pattern, #op_type__);                           \
    auto* out = CreateVarPDNode(pattern, {#op_type__}, {});                  \
    LINKS(x, op);                                                            \
    LINKS(y, op);                                                            \
    LINKS(z, op);                                                            \
    LINKS(op, out);                                                          \
    return out;                                                              \
  }

// simulate the pythoin DynamicRNN
TWO_ARG_OP(LoDTensorToArray, "lod_tensor_to_array");
TWO_ARG_OP(ArrayToLoDTensor, "array_to_lod_tensor");
ONE_ARG_OP(Increment, increment);

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

TWO_ARG_OP(LessThan, less_than);
TWO_ARG_OP(Equal, equal);
TWO_ARG_OP(ArrayRead, array_read);
ONE_ARG_OP(LoDRankTable, lod_rank_table);
ONE_ARG_OP(MaxSequenceLen, max_sequence_len);
TWO_ARG_OP(ReorderLodTensorByRank, reorder_lod_tensor_by_rank);
THREE_ARG_OP(ShrinkMemory, shrink_rnn_memory);
NO_ARG_OP(FillConstant, fill_constant);
TWO_ARG_OP(SequenceExpand, sequence_expand)
TWO_ARG_OP(Concat, concat);
TWO_ARG_OP(Mul, mul);
TWO_ARG_OP(ElementwiseAdd, elementwise_add);
ONE_ARG_OP(SequenceSoftmax, sequence_softmax);
TWO_ARG_OP(ElementwiseMul, elementwise_mul);
ONE_ARG_OP(SequencePool, sequence_pool);
ONE_ARG_OP(Sigmoid, sigmoid);
ONE_ARG_OP(Tanh, sigmoid);

PDNode* Sums(PDPattern* pattern, const std::vector<PDNode*>& xs) {
  auto* sum_op = CreateOpPNode(pattern, "sum");
  auto* out = CreateVarPDNode(pattern, {"sum"}, {});
  for (auto* x : xs) {
    LINKS(x, sum_op);
  }
  LINKS(sum_op, out);
  return out;
}

PDNode* FC(PDPattern* pattern, PDNode* input, bool has_bias = false) {
  std::vector<std::string> mul_out_outlink({"elementwise_add"});
  if (!has_bias) {
    mul_out_outlink.clear();
  }
  auto* w = CreateVarPDNode(pattern, {}, {"mul"});
  auto* mul_out = Mul(pattern, input, w);

  PDNode* res;

  if (has_bias) {
    auto* bias = CreateVarPDNode(pattern, {}, {"elementwise_add"});
    res = ElementwiseAdd(pattern, mul_out, bias);
  } else
    res = mul_out;
  return res;
}

PDNode* Linear(PDPattern* pattern, const std::vector<PDNode*>& xs,
               bool with_bias = true) {
  auto* sum_op = CreateOpPNode(pattern, "sum");
  auto* out = CreateVarPDNode(pattern, {"sum"}, {});
  std::vector<PDNode*> outs;
  for (auto* x : xs) {
    auto* y = FC(pattern, x, with_bias);
    outs.push_back(y);
    LINKS(y, sum_op);
  }
  LINKS(sum_op, out);
  return out;
}

// This pattern just match the external logic of DynamicRNN, its sub-block is
// not considered.
class DynamicRNN {
 public:
  DynamicRNN(PDPattern* external_p, PDPattern* subblock_p)
      : external_p_(external_p), subblock_p_(subblock_p) {
    step_index_ = CreateVarPDNode(external_p_, {"fill_constant"}, {});
    zero_idx_ = FillConstant(external_p_);
  }

  PDNode* StepInput(PDNode* x) {
    if (!lod_rank_table_) {
      lod_rank_table_ = LoDRankTable(external_p_, x);
      auto* max_seq_len = MaxSequenceLen(external_p_, lod_rank_table_);
      LessThan(external_p_, step_index_, max_seq_len);
    }

    auto* dynamic_rnn_input_array =
        LoDTensorToArray(external_p_, x, lod_rank_table_);
    return ArrayRead(external_p_, dynamic_rnn_input_array, step_index_);
  }

  PDNode* StaticInput(PDNode* x) {
    PADDLE_THROW("not implemented");
    return nullptr;
  }

  PDNode* Memory(PDNode* init_tensor, bool need_reorder = true) {
    if (init_tensor) {
      if (need_reorder) {
        init_tensor =
            ReorderLodTensorByRank(external_p_, init_tensor, lod_rank_table_);
      }

      // write to array
      auto* mem_array = CreateVarPDNode(external_p_, {"write_to_array"}, {});
      auto* write_to_array_op = CreateOpPNode(external_p_, "write_to_array");
      auto* pattern = external_p_;
      LINKS(init_tensor, write_to_array_op);
      LINKS(zero_idx_, write_to_array_op);

      auto* retv = ArrayRead(external_p_, mem_array, step_index_);
      ShrinkMemory(external_p_, retv, step_index_, lod_rank_table_);
      return retv;
    } else {
      PADDLE_THROW("not implemented");
    }
    return nullptr;
  }

  void UpdateMemory(PDNode* ex_mem, PDNode* new_mem) {
    LOG(WARNING) << "nothing will happen in UpdateMemory";
  }

  void Output(PDPattern* pattern, const std::vector<PDNode*>& outputs) {
    for (auto* x : outputs) {
      auto* outside_array = CreateVarPDNode(external_p_, {"array_write"}, {});
      ArrayWrite(external_p_, x, step_index_, outside_array);
    }
  }

 protected:
  PDNode* lod_rank_table_;
  PDNode* step_index_;
  PDNode* zero_idx_;

  PDPattern* external_p_;
  PDPattern* subblock_p_;
};

}  // namespace

class AttentionLSTM : public DynamicRNN {
 public:
  AttentionLSTM(PDPattern* external_p, PDPattern* subblock_p)
      : DynamicRNN(external_p, subblock_p) {}

  void Block(PDNode* data_repeat, PDNode* cell_init, PDNode* hidden_init) {
    auto* encoder_vec = StepInput(data_repeat);
    auto* cell_mem = Memory(cell_init, true);
    auto* hidden_mem = Memory(hidden_init, true);
    auto lstm_step_out =
        LstmStep(subblock_p_, encoder_vec, hidden_mem, cell_mem);
    PADDLE_ENFORCE_EQ(lstm_step_out.size(), 2UL);
    auto* h = lstm_step_out[0];
    auto* c = lstm_step_out[1];
    UpdateMemory(hidden_mem, h);
    UpdateMemory(cell_mem, c);
    Output(external_p_, {h});
  }

  std::vector<PDNode*> LstmStep(PDPattern* pattern, PDNode* x_t,
                                PDNode* hidden_t_pre, PDNode* cell_t_prev) {
    auto* cell_t_repeat = SequenceExpand(pattern, cell_t_prev, x_t);
    auto* attention_input = Concat(pattern, x_t, cell_t_repeat);
    auto* attention_fc = FC(pattern, attention_input, true);
    auto* attention_output = FC(pattern, attention_fc, true);
    auto* attention_output_actived = attention_output;
    auto* attention_weights =
        SequenceSoftmax(pattern, attention_output_actived);
    auto* scaled = ElementwiseMul(pattern, x_t, attention_weights);
    auto* context = SequencePool(pattern, scaled);

    auto* forget_gate =
        Sigmoid(pattern, Linear(pattern, {hidden_t_pre, context}, true));
    auto* input_gate =
        Sigmoid(pattern, Linear(pattern, {hidden_t_pre, context}, true));
    auto* output_gate =
        Sigmoid(pattern, Linear(pattern, {hidden_t_pre, context}, true));
    auto* cell_tilde =
        Tanh(pattern, Linear(pattern, {hidden_t_pre, context}, true));
    auto* tmp0 = ElementwiseMul(pattern, forget_gate, cell_t_prev);
    auto* tmp1 = ElementwiseMul(pattern, input_gate, cell_tilde);
    auto* cell_t = Sums(pattern, {tmp0, tmp1});
    auto* hidden_t =
        ElementwiseMul(pattern, output_gate, Tanh(pattern, cell_t));
    return {hidden_t, cell_t};
  }
};

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(attention_lstm_fuse_pass,
              paddle::framework::ir::AttentionLSTMFusePass);
