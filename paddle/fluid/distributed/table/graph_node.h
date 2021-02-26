#pragma once
#include "paddle/fluid/distributed/table/weighted_sampler.h"
#include<vector>
namespace paddle {
namespace distributed {
enum GraphNodeType{
	user=0,item=1,query=2,unknown=3
};
class GraphEdge: public WeightedObject{
public:
    double weight;
    uint64_t id;
    GraphNodeType type;
    GraphEdge(){
    }
    GraphEdge(uint64_t id, GraphNodeType type,double weight):weight(weight),id(id),type(type){
    }
};
class GraphNode{
public:
GraphNode(){
    sampler = NULL;
}
GraphNode(uint64_t id,GraphNodeType type,std::string feature):id(id),type(type),feature(feature),sampler(NULL){
}
virtual ~GraphNode() {}
static int enum_size,id_size,int_size,double_size;
uint64_t get_id(){
    return id;
}
void set_id(uint64_t id){
    this->id = id;
}
GraphNodeType get_graph_node_type(){
    return type;
}
void set_graph_node_type(GraphNodeType type){
    this->type = type;
}
void set_feature(std::string feature){
    this->feature = feature;
}
std::string get_feature(){
    return feature;
}
virtual int get_size();
virtual void build_sampler();
virtual void to_buffer(char* buffer);
virtual void recover_from_buffer(char* buffer);
virtual void add_edge(GraphEdge * edge){
    edges.push_back(edge);
}
static GraphNodeType get_graph_node_type(std::string &str){
      GraphNodeType type;
      if(str == "user")
            type = GraphNodeType::user;
       else if(str == "item")
            type = GraphNodeType::item;
       else if(str == "query")
          type = GraphNodeType:: query;
       else
           type = GraphNodeType::unknown;
      return type;   
}
std::vector<GraphEdge *> sample_k(int k){
    std::vector<GraphEdge *> v;
    if(sampler != NULL){
        auto res = sampler->sample_k(k);
        for(auto x: res){
            v.push_back((GraphEdge *)x);
        }
    }
    return v;
    
}
protected:
uint64_t id;
GraphNodeType type;
std::string feature;
WeightedSampler *sampler;
std::vector<GraphEdge *> edges;
};
}
}