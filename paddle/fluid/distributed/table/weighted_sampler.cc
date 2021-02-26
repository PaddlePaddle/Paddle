#include "paddle/fluid/distributed/table/weighted_sampler.h"
namespace paddle {
namespace distributed {
	void WeightedSampler::build(WeightedObject** v,int start,int end){
 		count = 0;
		if(start + 1 == end){
			left = right = NULL;
			weight = v[start]->get_weight();
			object = v[start];
			count = 1;

		} else {
			left = new WeightedSampler();
			right = new WeightedSampler();
			left->build(v,start, start + (end - start)/2);
			right->build(v,start + (end - start)/2, end);
			weight = left->weight + right->weight;
			count = left->count + right->count;
		}
	}
	vector<WeightedObject *> WeightedSampler::sample_k(int k){
		if(k > count){
			k = count;
		}
		vector<WeightedObject *> sample_result; 
		double subtract;
		unordered_map<WeightedSampler *,double> subtract_weight_map;
		unordered_map<WeightedSampler *,int> subtract_count_map;
		while(k--){
			double query_weight = rand() % 100000/100000.0;
			query_weight *= weight - subtract_weight_map[this];
			sample_result.push_back(sample(query_weight, subtract_weight_map,subtract_count_map, subtract));
		}
		return sample_result;

	}
	WeightedObject * WeightedSampler::sample(double query_weight, unordered_map<WeightedSampler *,double> &subtract_weight_map, unordered_map<WeightedSampler *,int> &subtract_count_map, double &subtract){
		if(left == NULL){
			subtract_weight_map[this] = weight;
			subtract = weight;
			subtract_count_map[this] = 1;
			return object;
		}
		int left_count = left->count - subtract_count_map[left];
		int right_count = right->count - subtract_count_map[right];			
		double left_subtract = subtract_weight_map[left];
		WeightedObject * return_id;
		if(right_count == 0 || left_count > 0 && left->weight - left_subtract >= query_weight){
			return_id = left->sample(query_weight, subtract_weight_map,subtract_count_map, subtract);
		} else {
			return_id = right->sample(query_weight - (left->weight - left_subtract),subtract_weight_map,subtract_count_map, subtract);
		}
		subtract_weight_map[this] += subtract;
		subtract_count_map[this]++;
		return return_id;

	}
}
}

