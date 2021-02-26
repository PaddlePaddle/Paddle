#include "paddle/fluid/distributed/table/graph_node.h"
#include<cstring>
namespace paddle {
namespace distributed {
int GraphNode:: enum_size = sizeof(int);
int GraphNode:: id_size = sizeof(uint64_t);
int GraphNode:: double_size = sizeof(double);
int GraphNode:: int_size = sizeof(int);
int GraphNode::get_size(){
    return feature.size() + id_size + enum_size + int_size;
}
void GraphNode::build_sampler(){
    sampler = new WeightedSampler();
    GraphEdge** arr = edges.data();
    sampler->build((WeightedObject **)arr, 0, edges.size());

}
void GraphNode:: to_buffer(char * buffer){
    int size = get_size();
    memcpy(buffer, &size, int_size);
    memcpy(buffer + int_size,feature.c_str(),feature.size());
    memcpy(buffer + int_size + feature.size(),&id,id_size);
    int int_state = type;
    memcpy(buffer + int_size + feature.size() + id_size,&int_state,enum_size);
}
void GraphNode:: recover_from_buffer(char* buffer){
    int size;
    memcpy(&size,buffer,int_size);
    int feature_size = size - id_size - enum_size - int_size;
    char str[feature_size + 1];
    memcpy(str,buffer + int_size,feature_size);
    str[feature_size] = '\0';
    feature = str;
    memcpy(&id,buffer + int_size + feature_size,id_size);
    int int_state;
    memcpy(&int_state,buffer + int_size + feature_size + id_size,enum_size);
    type = GraphNodeType(int_state);
}
}
}