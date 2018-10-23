
#ifdef _WIN32
#ifdef inference_icnet_EXPORTS
#define API_REFERENCE extern "C" __declspec(dllexport) 
#else
#define API_REFERENCE extern "C" __declspec(dllimport) 
#endif
#else
#define API_REFERENCE
#endif

//API_REFERENCE void * init_predictor();
//API_REFERENCE void destory_predictor(void *handle);
//API_REFERENCE void predict(void *handle, int n);

API_REFERENCE void * init_predictor(const char* prog_file,
	const char* param_file, const float fraction_of_gpu_memory,
	const bool use_gpu, const int device);
API_REFERENCE void predict(void* handle, float* input, const int channel, const int height,
	const int width, int64_t** output, int* output_length, int batch_size);
API_REFERENCE void destory_predictor(void *handle);
