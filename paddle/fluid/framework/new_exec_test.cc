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
#include "paddle/fluid/framework/new_exec.h"

#include <chrono>
#include <gperftools/profiler.h>


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
