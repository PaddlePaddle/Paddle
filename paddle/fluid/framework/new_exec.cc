#include <iostream>
#include <string>

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"

#include "paddle/fluid/pybind/pybind.h"

#include "paddle/fluid/platform/init.h"

#include <chrono>
#include <gperftools/profiler.h>

//USE_OP(fill_constant);
//USE_OP(elementwise_add);

using namespace std;

namespace paddle {
namespace framework {

class RuntimeInferShapeContext : public InferShapeContext {
 public:
  RuntimeInferShapeContext(const OperatorBase& op, const RuntimeContext& ctx)
      : op_(op), ctx_(ctx) {}

  bool HasInput(const std::string& name) const override {
    // has only one input
    const auto& ins = ctx_.inputs;
    auto it = ins.find(name);
    if (it == ins.end()) {
      return false;
    }
    const auto& in = it->second;
    if (in.size() == 0) return false;
    PADDLE_ENFORCE_EQ(
        in.size(), 1UL,
        platform::errors::InvalidArgument(
            "Input %s should not contain more than one inputs.", name));
    return in[0] != nullptr;
  }

  bool HasOutput(const std::string& name) const override {
    // has only one output
    const auto& outs = ctx_.outputs;
    auto it = outs.find(name);
    if (it == outs.end()) {
      return false;
    }
    const auto& out = it->second;
    if (out.size() == 0) {
      return false;
    }
    PADDLE_ENFORCE_EQ(
        out.size(), 1UL,
        platform::errors::InvalidArgument(
            "Output %s should not contain more than one outputs.", name));
    return out[0] != nullptr;
  }

  bool HasInputs(const std::string& name) const override {
    const auto& ins = ctx_.inputs;
    auto it = ins.find(name);
    if (it == ins.end() || it->second.empty()) {
      return false;
    }
    for (auto& input : it->second) {
      if (input == nullptr) {
        return false;
      }
    }
    return true;
  }

  bool HasOutputs(const std::string& name) const override {
    const auto& outs = ctx_.outputs;
    auto it = outs.find(name);
    if (it == outs.end() || it->second.empty()) {
      return false;
    }
    for (auto& output : it->second) {
      if (output == nullptr) {
        return false;
      }
    }
    return true;
  }

  AttrReader Attrs() const override { return AttrReader(op_.Attrs()); }

  std::vector<std::string> Inputs(const std::string& name) const override {
    return op_.Inputs(name);
  }

  std::vector<std::string> Outputs(const std::string& name) const override {
    return op_.Outputs(name);
  }

  std::string GetInputNameByIdx(size_t idx) const override {
    auto& op_proto =
        paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
    PADDLE_ENFORCE_LT(idx, op_proto->inputs().size(),
                      platform::errors::OutOfRange(
                          "The index should be less than the size of inputs of "
                          "operator %s, but got index is %d and size is %d",
                          op_.Type(), idx, op_proto->inputs().size()));
    return op_proto->inputs()[idx].name();
  }

  std::string GetOutputNameByIdx(size_t idx) const override {
    auto& op_proto =
        paddle::framework::OpInfoMap::Instance().Get(op_.Type()).proto_;
    PADDLE_ENFORCE_LT(
        idx, op_proto->outputs().size(),
        platform::errors::OutOfRange(
            "The index should be less than the size of outputs of "
            "operator %s, but got index is %d and size is %d",
            op_.Type(), idx, op_proto->outputs().size()));
    return op_proto->outputs()[idx].name();
  }

  void ShareDim(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) override {
    auto in_it = ctx_.inputs.find(in);
    auto out_it = ctx_.outputs.find(out);
    PADDLE_ENFORCE_NE(
        in_it, ctx_.inputs.end(),
        platform::errors::NotFound("Input %s does not exist.", in));
    PADDLE_ENFORCE_NE(
        out_it, ctx_.outputs.end(),
        platform::errors::NotFound("Output %s does not exist.", out));
    PADDLE_ENFORCE_LT(i, in_it->second.size(),
                      platform::errors::InvalidArgument(
                          "The index of input dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          in_it->second.size(), i));
    PADDLE_ENFORCE_LT(j, out_it->second.size(),
                      platform::errors::InvalidArgument(
                          "The index of output dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          out_it->second.size(), j));

    Variable* in_var = in_it->second[i];
    Variable* out_var = out_it->second[j];

    PADDLE_ENFORCE_EQ(
        in_var->Type(), out_var->Type(),
        platform::errors::InvalidArgument(
            "The type of input (%s) and output (%s) are inconsistent.", in,
            out));

    if (in_var->IsType<framework::SelectedRows>()) {
      auto& in_sele_rows = in_var->Get<framework::SelectedRows>();
      auto out_sele_rows = out_var->GetMutable<framework::SelectedRows>();
      out_sele_rows->mutable_value()->Resize(in_sele_rows.value().dims());
      out_sele_rows->set_rows(in_sele_rows.rows());
      out_sele_rows->set_height(in_sele_rows.height());
    } else if (in_var->IsType<framework::LoDTensor>()) {
      auto& in_lod_tensor = in_var->Get<framework::LoDTensor>();
      auto* out_lod_tensor = out_var->GetMutable<framework::LoDTensor>();
      out_lod_tensor->Resize(in_lod_tensor.dims());
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Currently, the input type of ShareDim only can be LoDTensor "
          "or SelectedRows."));
    }
  }

  void ShareAllLoD(const std::string& in,
                   const std::string& out) const override {
    auto in_it = ctx_.inputs.find(in);
    auto out_it = ctx_.outputs.find(out);
    PADDLE_ENFORCE_NE(in_it, ctx_.inputs.end(),
                      platform::errors::NotFound(
                          "Input [%s] found error in Op [%s]", in, op_.Type()));
    PADDLE_ENFORCE_NE(
        out_it, ctx_.outputs.end(),
        platform::errors::NotFound("Output [%s] found error in Op [%s]", out,
                                   op_.Type()));

    auto& in_var_list = in_it->second;
    auto& out_var_list = out_it->second;

    PADDLE_ENFORCE_EQ(
        in_var_list.size(), out_var_list.size(),
        platform::errors::PreconditionNotMet(
            "Op [%s]: Input var size should be equal with output var size",
            op_.Type()));

    auto& out_var_names = op_.Outputs(out);

    for (size_t i = 0; i < in_var_list.size(); ++i) {
      if (out_var_names[i] == framework::kEmptyVarName) {
        continue;
      }

      Variable* in_var = in_var_list[i];
      if (!in_var->IsType<LoDTensor>()) return;
      Variable* out_var = out_var_list[i];
      PADDLE_ENFORCE_EQ(out_var->IsType<LoDTensor>(), true,
                        platform::errors::PreconditionNotMet(
                            "The %d-th output of Output(%s) must be LoDTensor.",
                            i, out_var_names[i]));
      auto& in_tensor = in_var->Get<LoDTensor>();
      auto* out_tensor = out_var->GetMutable<LoDTensor>();
      out_tensor->set_lod(in_tensor.lod());
#ifdef PADDLE_WITH_MKLDNN
      if (in_tensor.layout() != DataLayout::kMKLDNN)
#endif
        out_tensor->set_layout(in_tensor.layout());
    }
  }

  void ShareLoD(const std::string& in, const std::string& out, size_t i = 0,
                size_t j = 0) const override {
    auto in_it = ctx_.inputs.find(in);
    auto out_it = ctx_.outputs.find(out);
    PADDLE_ENFORCE_NE(
        in_it, ctx_.inputs.end(),
        platform::errors::NotFound("Input %s does not exist.", in));
    PADDLE_ENFORCE_NE(
        out_it, ctx_.outputs.end(),
        platform::errors::NotFound("Output %s does not exist.", out));
    PADDLE_ENFORCE_LT(i, in_it->second.size(),
                      platform::errors::InvalidArgument(
                          "The index of input dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          in_it->second.size(), i));
    PADDLE_ENFORCE_LT(j, out_it->second.size(),
                      platform::errors::InvalidArgument(
                          "The index of output dimension is out of range, "
                          "excepted index less than %zu, but received %zu.",
                          out_it->second.size(), j));

    Variable* in_var = in_it->second.at(i);
    if (!in_var->IsType<LoDTensor>()) return;
    Variable* out_var = out_it->second.at(j);
    PADDLE_ENFORCE_EQ(
        out_var->IsType<LoDTensor>(), true,
        platform::errors::InvalidArgument(
            "The %zu-th output of Output(%s) must be LoDTensor.", j, out));
    auto& in_tensor = in_var->Get<LoDTensor>();
    auto* out_tensor = out_var->GetMutable<LoDTensor>();
    out_tensor->set_lod(in_tensor.lod());

// TODO(dzhwinter) : reuse ShareLoD in most operators.
// Need to call ShareLayout explicitly in sequence related ops.
// Shall we have a better method to shared info between in/out Tensor?
#ifdef PADDLE_WITH_MKLDNN
    // Fix me: ugly workaround below
    // Correct solution:
    //    set_layout() should NOT be called here (i.e. ShareLoD). Instead,
    //    layout of output tensor should be set "manually" in Compute()
    //    of each OPKernel. The reason layout should NOT be shared between
    //    input and output "automatically" (now by InferShape()->ShareLoD())
    //    is that layout transform may occur after InferShape().
    // Workaround:
    //    Skip set_layout() when input layout is kMKLDNN
    //    This is to avoid kMKLDNN is populated wrongly into a non-MKLDNN
    //    OPKernel. In all MKLDNN OPkernel, set_layout(kMKLDNN) should be called
    //    in Compute()
    if (in_tensor.layout() != DataLayout::kMKLDNN)
#endif
      out_tensor->set_layout(in_tensor.layout());
  }

  int32_t GetLoDLevel(const std::string& in, size_t i = 0) const override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GetLoDLevel is only used in compile time. The calculation of "
        "output's actual lod is different among operators so that should be "
        "set in the runtime kernel."));
  }

  void SetLoDLevel(const std::string& out, int32_t lod_level,
                   size_t j = 0) const override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "SetLoDLevel is only used in compile time. The calculation of "
        "output's actual lod is different among operators so that should be "
        "set in the runtime kernel."));
  }

  bool IsRuntime() const override { return true; }

  // TODO(paddle-dev): Can this be template?
  std::vector<InferShapeVarPtr> GetInputVarPtrs(
      const std::string& name) override {
    const std::vector<Variable*>& vars = InputVars(name);
    std::vector<InferShapeVarPtr> res;
    res.reserve(vars.size());
    res.insert(res.begin(), vars.begin(), vars.end());
    return res;
  }

  std::vector<InferShapeVarPtr> GetOutputVarPtrs(
      const std::string& name) override {
    const std::vector<Variable*>& vars = OutputVars(name);
    std::vector<InferShapeVarPtr> res;
    res.reserve(vars.size());
    res.insert(res.begin(), vars.begin(), vars.end());
    return res;
  }

  DDim GetInputDim(const std::string& name) const override {
    const std::vector<Variable*>& vars = InputVars(name);
    PADDLE_ENFORCE_EQ(
        vars.size(), 1UL,
        platform::errors::InvalidArgument(
            "Input(%s) should hold one element, but now it holds %zu elements.",
            name, vars.size()));
    return this->GetDim(vars[0]);
  }

  std::vector<DDim> GetInputsDim(const std::string& name) const override {
    const std::vector<Variable*>& vars = InputVars(name);
    return GetDims(vars);
  }

  std::vector<proto::VarType::Type> GetInputsVarType(
      const std::string& name) const override {
    return GetVarTypes(InputVars(name));
  }

  std::vector<proto::VarType::Type> GetOutputsVarType(
      const std::string& name) const override {
    return GetVarTypes(OutputVars(name));
  }

  void SetOutputDim(const std::string& name, const DDim& dim) override {
    //cerr << "set out dim" << endl;
    auto& vars = OutputVars(name);
    PADDLE_ENFORCE_EQ(
        vars.size(), 1UL,
        platform::errors::InvalidArgument("Output(%s) should hold one element, "
                                          "but now it holds %zu elements.",
                                          name, vars.size()));
    SetDim(vars[0], dim);
  }

  void SetOutputsDim(const std::string& name,
                     const std::vector<DDim>& dims) override {
    auto& vars = OutputVars(name);
    SetDims(vars, dims);
  }

 protected:
  DDim GetDim(Variable* var) const {
    PADDLE_ENFORCE_NOT_NULL(
        var, platform::errors::InvalidArgument("Input variable is nullptr."));
    if (var->IsType<LoDTensor>()) {
      return var->Get<LoDTensor>().dims();
    } else if (var->IsType<SelectedRows>()) {
      return var->Get<SelectedRows>().GetCompleteDims();
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Only LoDTensor or SelectedRows support 'GetDim', but input "
          "Variable's type is %s.",
          ToTypeName(var->Type())));
    }
  }

  std::vector<DDim> GetDims(const std::vector<Variable*>& vars) const {
    std::vector<DDim> ret;
    ret.reserve(vars.size());
    std::transform(vars.begin(), vars.end(), std::back_inserter(ret),
                   [this](Variable* var) { return this->GetDim(var); });
    return ret;
  }

  std::vector<DDim> GetRepeatedDims(const std::string& name) const override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "GetRepeatedDims method only ban be used in compile time."));
  }

  void SetDim(Variable* var, const DDim& dim) {
   
    if (var->IsType<LoDTensor>()) {
   
      var->GetMutable<LoDTensor>()->Resize(dim);
    } else if (var->IsType<SelectedRows>()) {
      var->GetMutable<SelectedRows>()->set_height(dim[0]);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Variable type error, expect LoDTensor or SelectedRows, but received "
          "(%s).",
          ToTypeName(var->Type())));
    }
  }

  void SetDims(const std::vector<Variable*>& vars,
               const std::vector<DDim>& dims) {
    size_t length = vars.size();
    PADDLE_ENFORCE_EQ(length, dims.size(),
                      platform::errors::InvalidArgument(
                          "The number of input variables do not match the "
                          "number of input dimensions, the number of variables "
                          "is %zu, the number of dimensions is %zu.",
                          length, dims.size()));
    for (size_t i = 0; i < length; ++i) {
      if (vars[i] == nullptr) {
        continue;
      }
      SetDim(vars[i], dims[i]);
    }
  }

  void SetRepeatedDims(const std::string& name,
                       const std::vector<DDim>& dims) override {
    PADDLE_THROW(platform::errors::PreconditionNotMet(
        "SetRepeatedDims method only can be used in compile time."));
  }

  std::vector<proto::VarType::Type> GetVarTypes(
      const std::vector<Variable*>& vars) const {
    std::vector<proto::VarType::Type> retv;
    retv.resize(vars.size());
    std::transform(vars.begin(), vars.end(), retv.begin(),
                   std::bind(std::mem_fn(&RuntimeInferShapeContext::GetVarType),
                             this, std::placeholders::_1));
    return retv;
  }

  proto::VarType::Type GetVarType(Variable* var) const {
    return ToVarType(var->Type());
  }

 private:
  const std::vector<Variable*>& InputVars(const std::string& name) const {
    auto it = ctx_.inputs.find(name);
    PADDLE_ENFORCE_NE(
        it, ctx_.inputs.end(),
        platform::errors::NotFound(
            "Operator (%s) does not have the input (%s).", op_.Type(), name));
    return it->second;
  }

  const std::vector<Variable*>& OutputVars(const std::string& name) const {
    auto it = ctx_.outputs.find(name);
    PADDLE_ENFORCE_NE(
        it, ctx_.outputs.end(),
        platform::errors::NotFound(
            "Operator (%s) does not have the outputs (%s).", op_.Type(), name));
    return it->second;
  }

  const OperatorBase& op_;
  const RuntimeContext& ctx_;
};



framework::ProgramDesc load_from_file( const std::string& file_name )
{
  std::ifstream fin(file_name, std::ios::in | std::ios::binary);
  fin.seekg(0, std::ios::end);
  std::string buffer(fin.tellg(), ' ');
  fin.seekg(0, std::ios::beg);
  fin.read(&buffer[0], buffer.size());
  fin.close();

  ProgramDesc program_desc( buffer );
  return program_desc;
}


struct VariableScope
{
    std::vector< std::unique_ptr<Variable> > var_list;
    std::map<std::string, int> name2id;
};


struct OpFuncNode{

    //int unsed;
    std::map< std::string, std::vector<int> > input_index;
    std::map< std::string, std::vector<int> > output_index;
    
    using OpKernelFunc = std::function<void(const ExecutionContext&)>;
    OpKernelFunc kernel_func_;
};


void build_variable_scope( const framework::ProgramDesc& pdesc, VariableScope* var_scope )
{
  auto& global_block = pdesc.Block(0);
  
  
  for (auto& var : global_block.AllVars()) {
      if (var->Name() == framework::kEmptyVarName) {
        continue;
      }
      //cerr << "var name "  << var->Name() << endl;  

      if ( var_scope->name2id.find( var->Name() ) == var_scope->name2id.end() )
      {
          var_scope->name2id[ var->Name() ] = var_scope->var_list.size();
      }
      
      auto v = new Variable();
      v->GetMutable<LoDTensor>();
      var_scope->var_list.push_back(std::unique_ptr<Variable>(v));
  }
}

void build_op_func_list( const framework::ProgramDesc& pdesc, std::vector<std::unique_ptr<OperatorBase> >& op_list, std::vector<OpFuncNode>& vec_func_list, const VariableScope& var_scope )
{
    auto &global_block = pdesc.Block( 0 );

    for ( auto& op : global_block.AllOps() )
    { 
        //cerr << op->Type() << endl;
        //bool debug = op->Type() == "softmax_with_cross_entropy_grad";
        bool debug = false;
        op_list.push_back( OpRegistry::CreateOp(*op) );
        //cerr << "create op" << endl;
        auto* op_base = op_list.back().get();

        auto input_names = op->Inputs();
        auto output_names = op->Outputs();
        
        OpFuncNode op_func_node;

        VariableValueMap ins_map;
        std::map< std::string, std::vector<int> > ins_name2id;
        for( auto& var_name_item : input_names)
        {
            std::vector<Variable*> input_vars;
            std::vector<int> vec_ids;
            input_vars.reserve(var_name_item.second.size());
            for (auto& var_name : var_name_item.second) {
                auto it = var_scope.name2id.find( var_name );
                assert( it != var_scope.name2id.end() );
                input_vars.push_back( var_scope.var_list[ it->second].get());
                vec_ids.push_back( it->second );
            }
            ins_map[ var_name_item.first ] = input_vars;
            ins_name2id[ var_name_item.first ] = vec_ids;

        }
        if (debug  ) cerr << "1" << endl;
        
        VariableValueMap outs_map;
        std::map<std::string, std::vector<int> > outs_name2id;
        for( auto& var_name_item : output_names )
        {
            std::vector<Variable*> output_vars;
            std::vector<int> vec_ids;
            output_vars.reserve(var_name_item.second.size());
            for (auto& var_name : var_name_item.second) {
                auto it = var_scope.name2id.find( var_name );
                assert( it != var_scope.name2id.end() );
                //cerr << it->second << "\t" << var_scope.var_list.size() << endl;
                output_vars.push_back( var_scope.var_list[ it->second].get() );
                vec_ids.push_back( it->second );
            } 
            outs_map[ var_name_item.first ] = output_vars;
            //cerr << ToTypeName(output_vars[0]->Type() ) << endl;
            outs_name2id[ var_name_item.first ] = vec_ids;            
        }

        op_func_node.input_index = ins_name2id;
        op_func_node.output_index = outs_name2id;
        RuntimeContext runtime_context( {}, {});
        runtime_context.inputs.swap( ins_map );
        runtime_context.outputs.swap(  outs_map );
        //cerr << "create runtime context" << endl;
        RuntimeInferShapeContext infer_shape_ctx(*op_base, runtime_context);
        static_cast<const framework::OperatorWithKernel*>(op_base)->InferShape( &infer_shape_ctx );
        //cerr << "fin infer shape" << endl;
        auto& all_op_kernels = OperatorWithKernel::AllOpKernels();
        auto kernels_iter = all_op_kernels.find(op->Type() );
        PADDLE_ENFORCE_NE(
            kernels_iter, all_op_kernels.end(),
            platform::errors::Unavailable(
                "There are no kernels which are registered in the %s operator.",
                op->Type() ));
        
        //cerr << "create kernel" << endl;
        using OpKernelFunc = std::function<void(const ExecutionContext&)>;
        using OpKernelMap =
             std::unordered_map<OpKernelType, OpKernelFunc, OpKernelType::Hash>;
        if (debug  ) cerr << "2" << endl;
        OpKernelMap& kernels = kernels_iter->second;
        //auto place = platform::CPUPlace();
        auto place = platform::CUDAPlace(0);
        platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
        auto* dev_ctx = pool.Get(place);
        Scope scope;
        auto exec_ctx = ExecutionContext(*op_base, scope, *dev_ctx, runtime_context );
        if (debug  ) cerr << "21" << endl;
        auto expected_kernel_key = dynamic_cast<const framework::OperatorWithKernel*>(op_base)->GetExpectedKernelType( exec_ctx );
        if (debug  ) cerr << "22" << endl;
        //cerr << "22" << endl;
        auto kernel_iter = kernels.find(expected_kernel_key);

        if (debug  ) cerr << "3" << endl;
        op_func_node.kernel_func_ = OpKernelFunc(kernel_iter->second);
        if (debug  ) cerr << "3-1" << endl;
        op_func_node.kernel_func_(  exec_ctx );
        vec_func_list.push_back( op_func_node );
        if (debug  ) cerr << "5" << endl;
    }
    
}



void exec_op_func_list( const std::vector<OpFuncNode>& vec_func_list, std::vector<std::unique_ptr<OperatorBase>>& op_list, const VariableScope& var_scope)
{
    for( size_t i = 0; i < vec_func_list.size(); ++i )    
    {
        auto& func_node = vec_func_list[i];
        auto op_base = op_list[i].get();
        // build runtime cost
        VariableValueMap ins_map;        
        for( auto& var_name_item : func_node.input_index)
        {
            std::vector<Variable*> input_vars;
            
            input_vars.reserve(var_name_item.second.size());
            for (auto& id : var_name_item.second) {                
                input_vars.emplace_back( var_scope.var_list[ id ].get() );                
            }
            ins_map.emplace( var_name_item.first, std::move(input_vars) );            
        }

        VariableValueMap outs_map;        
        for( auto& var_name_item : func_node.output_index)
        {
            std::vector<Variable*> out_vars;
            
            out_vars.reserve(var_name_item.second.size());
            for (auto& id : var_name_item.second) {                
                out_vars.emplace_back( var_scope.var_list[ id ].get());                 
            }            
            outs_map.emplace( var_name_item.first, std::move( out_vars ) );
        }

        RuntimeContext runtime_context( {}, {});
        runtime_context.inputs.swap( ins_map );
        runtime_context.outputs.swap(  outs_map );

        RuntimeInferShapeContext infer_shape_ctx(*(op_list[i].get()), runtime_context);
        dynamic_cast<const framework::OperatorWithKernel*>(op_base)->InferShape( &infer_shape_ctx );


        platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
        //auto place = platform::CPUPlace();
        auto place = platform::CUDAPlace(0);
        auto* dev_ctx = pool.Get(place);
        Scope scope;
        
        auto exec_context = ExecutionContext(*op_base, scope, *dev_ctx, runtime_context );

        func_node.kernel_func_( exec_context );
    }
}



}
}


int main()
{   
    paddle::framework::InitDevices();
    paddle::framework::VariableScope global_scope;

    
    {
        auto test_prog = paddle::framework::load_from_file( "lm_startup_program");
        paddle::framework::build_variable_scope( test_prog, &global_scope );


        std::vector<paddle::framework::OpFuncNode> vec_func_list;
        std::vector<std::unique_ptr<paddle::framework::OperatorBase>> op_list;
        paddle::framework::build_op_func_list( test_prog, op_list, vec_func_list, global_scope);

        paddle::framework::exec_op_func_list( vec_func_list, op_list, global_scope ); 
    }

    cerr << "run main" << endl;
    auto main_prog = paddle::framework::load_from_file( "lm_main_program");

    paddle::framework::build_variable_scope( main_prog, &global_scope );


    std::vector<paddle::framework::OpFuncNode> vec_main_func_list;
    std::vector<std::unique_ptr<paddle::framework::OperatorBase>> op_main_list;
    paddle::framework::build_op_func_list( main_prog, op_main_list, vec_main_func_list, global_scope);

    auto start = std::chrono::steady_clock::now();
    ProfilerStart("new_executor.prof");
    for ( size_t i = 0; i < 2320; ++i )
    {
        if( i % 200 == 0)
        {   
            cerr << i << endl;
        }
        paddle::framework::exec_op_func_list( vec_main_func_list, op_main_list, global_scope ); 
    }
    ProfilerStop();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end-start;

    cerr << "time cost " << diff.count() << endl;


    return 1;

}
