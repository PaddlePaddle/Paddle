#include <algorithm>
#include <memory>
#include <vector>
#include <iostream>


#include "paddle/fluid/imperative/tracer.h"
#include "paddle/fluid/imperative/layer.h"

#include "paddle/fluid/pybind/pybind.h"

#include <gperftools/profiler.h>

#include "paddle/fluid/platform/init.h"

#include <chrono>
#include "paddle/fluid/platform/cuda_profiler.h"

using namespace std;

//USE_OP(concat);
//USE_OP(sigmoid);
//USE_OP(elementwise_mul);




namespace paddle {
namespace imperative {


using namespace framework;
//TEST(Tracer, Trace) {
void f_test(){
    cerr << "test by defalut" << endl;
    LOG(ERROR) << "test new !!!";
    
    
    paddle::framework::InitDevices(true);

    //auto place = platform::CPUPlace();

    //cerr << "get palce " << endl;
    auto place = platform::CUDAPlace(0);

    int seq_len = 100;
    int batch_size = 20;
    int hidden_size = 200;
    //  init input var
    vector< std::shared_ptr<VarBase> > vec_input_var;
    for( int i = 0; i < seq_len; ++i )
    {   
        vector<int64_t> vec_init_shape = { batch_size, hidden_size};
    
        std::shared_ptr<VarBase> var_init( new VarBase( "in", paddle::framework::proto::VarType::FP32,  vec_init_shape, place, false, true, true) );
        // set value here
        vec_input_var.push_back( var_init );
    }



    
    vector<OpBase*> vec_op_list;
    vector< VarBasePtrMap > vec_var_map_input;
    vector< VarBasePtrMap > vec_var_map_output;
    vector< framework::AttributeMap > vec_attr_map;
    
    vector<int64_t> vec_pre_shape = { batch_size, hidden_size };

    std::shared_ptr<VarBase> pre_hidden( new VarBase( "pre_hidden", paddle::framework::proto::VarType::FP32,  vec_pre_shape, place, false, true, true) );
    std::shared_ptr<VarBase> pre_cell( new VarBase( "pre_cell", paddle::framework::proto::VarType::FP32,  vec_pre_shape, place, false, true, true) );

    std::shared_ptr<VarBase> weight( new VarBase( "weight", paddle::framework::proto::VarType::FP32,  {hidden_size * 2, hidden_size * 4}, place, false, true, true) );
    std::shared_ptr<VarBase> bias( new VarBase( "bias", paddle::framework::proto::VarType::FP32,  { hidden_size * 4 }, place, false, true, true) );
    //need to set real value here

    auto tracer = new Tracer();
    auto float_type = paddle::framework::proto::VarType::FP32;
    
    auto start = std::chrono::steady_clock::now();
    int max_loop = 1000;

    //StartProfile();
    //ProfilerStart("test.prof");

    for( int loop = 0; loop < max_loop; ++loop)
    {
        for ( int i = 0; i < seq_len; ++i)
        {
            // concat
            std::shared_ptr<OpBase> concat_op( new OpBase("concat") );        
            std::shared_ptr<VarBase> var_concat_output( new VarBase( "concat_out", float_type,  {}, place, false, true, true)  );
            vector< std::shared_ptr<VarBase> > vec_concat_in = { pre_hidden, vec_input_var[i] };
            vector< std::shared_ptr<VarBase> > vec_concat_out = { var_concat_output};
            VarBasePtrMap concat_map_in, concat_map_out;
            concat_map_in["X"]  = vec_concat_in;
            concat_map_out["Out"] = vec_concat_out;
            framework::AttributeMap concat_att_map(10);
            concat_att_map["axis"] = 1;

            tracer->Trace( concat_op.get(), concat_map_in, &concat_map_out, concat_att_map, place, false);
            
            // matmul
            std::shared_ptr<OpBase> matmul_op( new OpBase("matmul") );        
            std::shared_ptr<VarBase> var_matmul_output( new VarBase( "matmul_out", float_type,  {}, place, false, true, true)  );
            vector< std::shared_ptr<VarBase> > vec_matmul_in_w = { weight };
            vector< std::shared_ptr<VarBase> > vec_matmul_out = { var_matmul_output};
            VarBasePtrMap matmul_map_in, matmul_map_out;
            matmul_map_in["X"] = vec_concat_out;
            matmul_map_in["Y"] = vec_matmul_in_w;
            matmul_map_out["Out"] = vec_matmul_out;
            framework::AttributeMap matmul_att_map;     
            
            tracer->Trace( matmul_op.get(), matmul_map_in, &matmul_map_out, matmul_att_map, place, false);

            // ele_add
            std::shared_ptr<OpBase> ele_add_op( new OpBase("elementwise_add") );        
            std::shared_ptr<VarBase> var_ele_add_output( new VarBase( "ele_add_out", float_type,  {}, place, false, true, true)  );
            vector< std::shared_ptr<VarBase> > vec_ele_add_in_y = { bias };
            vector< std::shared_ptr<VarBase> > vec_ele_add_out = { var_ele_add_output };
            VarBasePtrMap ele_add_map_in, ele_add_map_out;
            ele_add_map_in["X"] = vec_matmul_out;
            ele_add_map_in["Y"] = vec_ele_add_in_y;
            ele_add_map_out["Out"] = vec_ele_add_out;
            framework::AttributeMap ele_add_att_map(10);
            ele_add_att_map["axis"] = -1;

            tracer->Trace( ele_add_op.get(), ele_add_map_in, &ele_add_map_out, ele_add_att_map, place, false);

            // split
            std::shared_ptr<OpBase> split_op( new OpBase("split") );        
            std::shared_ptr<VarBase> var_split_output_i( new VarBase( "split_out_0", float_type,  {}, place, false, true, true)  );
            std::shared_ptr<VarBase> var_split_output_j( new VarBase( "split_out_1", float_type,  {}, place, false, true, true)  );
            std::shared_ptr<VarBase> var_split_output_f( new VarBase( "split_out_2", float_type,  {}, place, false, true, true)  );
            std::shared_ptr<VarBase> var_split_output_o( new VarBase( "split_out_3", float_type,  {}, place, false, true, true)  );        
            vector< std::shared_ptr<VarBase> > vec_split_out = { var_split_output_i, var_split_output_j, var_split_output_f, var_split_output_o };
            VarBasePtrMap split_map_in, split_map_out;
            split_map_in["X"] = vec_ele_add_out;        
            split_map_out["Out"] = vec_split_out;
            framework::AttributeMap split_att_map(10);
            split_att_map["num"] = 4;
            split_att_map["axis"] = 1;

            tracer->Trace( split_op.get(), split_map_in, &split_map_out, split_att_map, place, false);

            // sigmoid forget
            std::shared_ptr<OpBase> forget_sig_op( new OpBase("sigmoid") );        
            std::shared_ptr<VarBase> var_forget_sig_output( new VarBase( "forget_sig", float_type,  {}, place, false, true, true)  );        
            vector< std::shared_ptr<VarBase> > vec_forget_sig_out = { var_forget_sig_output };
            VarBasePtrMap forget_sig_map_in, forget_sig_map_out;
            forget_sig_map_in["X"] = { var_split_output_f };        
            forget_sig_map_out["Out"] = vec_forget_sig_out;
            framework::AttributeMap forget_sig_att_map;        

            tracer->Trace( forget_sig_op.get(), forget_sig_map_in, &forget_sig_map_out, forget_sig_att_map, place, false);

            // ele mul forget and pre_cell
            std::shared_ptr<OpBase> ele_mul_forget_pre_cell_op( new OpBase("elementwise_mul") );        
            std::shared_ptr<VarBase> var_forget_mul_pre_cell_output( new VarBase( "forget_mul_pre_cell", float_type,  {}, place, false, true, true)  );        
            vector< std::shared_ptr<VarBase> > vec_forget_mul_pre_cell_out = { var_forget_mul_pre_cell_output };
            VarBasePtrMap forget_mul_pre_cell_map_in, forget_mul_pre_cell_map_out;
            forget_mul_pre_cell_map_in["X"] = { pre_cell };        
            forget_mul_pre_cell_map_in["Y"] = vec_forget_sig_out;
            forget_mul_pre_cell_map_out["Out"] = vec_forget_mul_pre_cell_out;
            framework::AttributeMap forget_mul_pre_cell_att_map;        

            tracer->Trace( ele_mul_forget_pre_cell_op.get(), forget_mul_pre_cell_map_in, &forget_mul_pre_cell_map_out, forget_mul_pre_cell_att_map, place, false);

            // sigmoid input
            std::shared_ptr<OpBase> input_sig_op( new OpBase("sigmoid") );        
            std::shared_ptr<VarBase> var_input_sig_output( new VarBase( "input_sig", float_type,  {}, place, false, true, true)  );        
            vector< std::shared_ptr<VarBase> > vec_input_sig_out = { var_input_sig_output };
            VarBasePtrMap input_sig_map_in, input_sig_map_out;
            input_sig_map_in["X"] = { var_split_output_i };        
            input_sig_map_out["Out"] = vec_input_sig_out;
            framework::AttributeMap input_sig_att_map;        

            tracer->Trace( input_sig_op.get(), input_sig_map_in, &input_sig_map_out, input_sig_att_map, place, false);

            // tanh j 
            std::shared_ptr<OpBase> tanh_j_op( new OpBase("tanh") );        
            std::shared_ptr<VarBase> var_tanh_j_output( new VarBase( "tanh_j", float_type,  {}, place, false, true, true)  );        
            vector< std::shared_ptr<VarBase> > vec_tanh_j_out = { var_tanh_j_output };
            VarBasePtrMap tanh_j_map_in, tanh_j_map_out;
            tanh_j_map_in["X"] = { var_split_output_j };        
            tanh_j_map_out["Out"] = vec_tanh_j_out;
            framework::AttributeMap tanh_j_att_map;        

            tracer->Trace( tanh_j_op.get(), tanh_j_map_in, &tanh_j_map_out, tanh_j_att_map, place, false);

            // ele mul j and input
            std::shared_ptr<OpBase> ele_mul_j_input_op( new OpBase("elementwise_mul") );        
            std::shared_ptr<VarBase> var_input_j_output( new VarBase( "input_j", float_type,  {}, place, false, true, true)  );        
            vector< std::shared_ptr<VarBase> > vec_input_j_out = { var_input_j_output };
            VarBasePtrMap input_j_map_in, input_j_map_out;
            input_j_map_in["X"] = vec_input_sig_out;        
            input_j_map_in["Y"] = vec_tanh_j_out;
            input_j_map_out["Out"] = vec_input_j_out;
            framework::AttributeMap input_j_att_map;        

            tracer->Trace( ele_mul_j_input_op.get(), input_j_map_in, &input_j_map_out, input_j_att_map, place, false);

            // ele add new_cell
            std::shared_ptr<OpBase> new_cell_op( new OpBase("elementwise_add") );        
            std::shared_ptr<VarBase> var_new_cell_output( new VarBase( "new_cell", float_type,  {}, place, false, true, true)  );        
            vector< std::shared_ptr<VarBase> > vec_new_cell_out = { var_new_cell_output };
            VarBasePtrMap new_cell_map_in, new_cell_map_out;
            new_cell_map_in["X"] = vec_forget_mul_pre_cell_out;        
            new_cell_map_in["Y"] = vec_input_j_out;
            new_cell_map_out["Out"] = vec_new_cell_out;
            framework::AttributeMap new_cell_att_map;        

            tracer->Trace( new_cell_op.get(), new_cell_map_in, &new_cell_map_out, new_cell_att_map, place, false);

            // tanh cell
            std::shared_ptr<OpBase> tanh_new_cell_op( new OpBase("tanh") );        
            std::shared_ptr<VarBase> var_tanh_new_cell_output( new VarBase( "tanh_new_cell", float_type,  {}, place, false, true, true)  );        
            vector< std::shared_ptr<VarBase> > vec_tanh_new_cell_out = { var_tanh_new_cell_output };
            VarBasePtrMap tanh_new_cell_map_in, tanh_new_cell_map_out;
            tanh_new_cell_map_in["X"] = vec_new_cell_out;        
            tanh_new_cell_map_out["Out"] = vec_tanh_new_cell_out;
            framework::AttributeMap tanh_new_cell_att_map;        

            tracer->Trace( tanh_new_cell_op.get(), tanh_new_cell_map_in, &tanh_new_cell_map_out, tanh_new_cell_att_map, place, false);

            // output gate
            // sigmoid output
            std::shared_ptr<OpBase> output_sig_op( new OpBase("sigmoid") );        
            std::shared_ptr<VarBase> var_output_sig_output( new VarBase( "output_sig", float_type,  {}, place, false, true, true)  );        
            vector< std::shared_ptr<VarBase> > vec_output_sig_out = { var_output_sig_output };
            VarBasePtrMap output_sig_map_in, output_sig_map_out;
            output_sig_map_in["X"] = { var_split_output_o };        
            output_sig_map_out["Out"] = vec_output_sig_out;
            framework::AttributeMap output_sig_att_map;        

            tracer->Trace( output_sig_op.get(), output_sig_map_in, &output_sig_map_out, output_sig_att_map, place, false);

            // ele mul tanh cell and output gate
            std::shared_ptr<OpBase> new_hidden_op( new OpBase("elementwise_mul") );        
            std::shared_ptr<VarBase> var_new_hidden_output( new VarBase( "new_hidden", float_type,  {}, place, false, true, true)  );        
            vector< std::shared_ptr<VarBase> > vec_new_hidden_out = { var_new_hidden_output };
            VarBasePtrMap new_hidden_map_in, new_hidden_map_out;
            new_hidden_map_in["X"] = vec_tanh_new_cell_out;        
            new_hidden_map_in["Y"] = vec_output_sig_out;
            new_hidden_map_out["Out"] = vec_new_hidden_out;
            framework::AttributeMap new_hidden_att_map;        

            tracer->Trace( new_hidden_op.get(), new_hidden_map_in, &new_hidden_map_out, new_hidden_att_map, place, false);
        }
    }

    //ProfilerStop();
    //ProfilerFlush();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end-start;

    cerr << "time cost " << diff.count() << endl;
    /*
    OpBase* op_base_1 = new OpBase("one_hot");

    vector<int64_t> var_shape = {5, 1};
    std::shared_ptr<VarBase> var_in( new VarBase( "input", paddle::framework::proto::VarType::INT32,  var_shape, platform::CPUPlace(), false, true, true) );

    std::shared_ptr<VarBase> var_out( new VarBase( "output", paddle::framework::proto::VarType::FP32,  var_shape, platform::CPUPlace(), false, true, false) );

    vector<std::shared_ptr<VarBase>> vec_in = {var_in};
    vector<std::shared_ptr<VarBase>> vec_out = { var_out };

    VarBasePtrMap map_in;
    map_in["X"] = vec_in;
    VarBasePtrMap map_out;
    map_out["Out"] = vec_out;
   

    framework::AttributeMap att_map;

    att_map["depth"] =  10;
    */
    
    /*
    auto start = std::chrono::steady_clock::now();
    for( int loop = 0; loop < 200; ++loop )
    {
        for ( size_t i = 0; i < vec_op_list.size(); ++i )
        {
            //cerr << vec_op_list[i]->type_ << endl;
            tracer->Trace( vec_op_list[i], vec_var_map_input[i], &vec_var_map_output[i], vec_attr_map[i], place, false );
        
            //cerr << vec_var_map_output[i]["Out"][0]->var_->Get<framework::LoDTensor>() << endl; 
            //cerr << i << endl;
        }
    }
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = end-start;

    cerr << "time cost " << diff.count() << endl;
    */
    
    //cerr << "init finished" << endl;
    //cerr << var_in->var_->Get<framework::LoDTensor>() << endl;
    //cerr << var_out->var_->Get<framework::LoDTensor>() << endl;

}
/*
void main( int arg, char** args)
{
    f_test();
}
*/
}
}

int main( int arg, char** args )
{
    cerr << "test" << endl;
    
    using namespace paddle;
    using namespace imperative;

    paddle::imperative::f_test();
    
    /*
    ProfilerStart("test.prof");

    map<int, double> map_test;
    int j = 0;
    for ( int i = 0; i < 1000000; ++i )
    {
        j = exp(i );
        map_test[i] = j;
    }
    ProfilerStop();
    */
    return 0;
}
