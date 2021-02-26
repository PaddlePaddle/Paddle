#pragma once
#include<vector>
#include<unordered_map>
#include<ctime>
using namespace std;
namespace paddle {
namespace distributed {
class WeightedObject{
public:
	WeightedObject(){

	}
	virtual ~WeightedObject(){

	}
	virtual unsigned long long get_id(){
		return id;
	}
	virtual double get_weight(){
		return weight;
	}
	
	virtual void set_id(unsigned long long id){
		this->id = id;
	}
	virtual void set_weight(double weight){
		this->weight = weight;
	}	
private:
	unsigned long long id;
	double weight;

};

class WeightedSampler{
public:
	WeightedSampler* left,*right;
	WeightedObject* object;
	int count;
	double weight;
	void build(WeightedObject** v,int start,int end);
	vector<WeightedObject*> sample_k(int k);
private:
	WeightedObject* sample(double query_weight, unordered_map<WeightedSampler *,double> &subtract_weight_map, unordered_map<WeightedSampler *,int> &subtract_count_map, double &subtract);
};
}
}