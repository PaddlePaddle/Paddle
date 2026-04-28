#define CHECK_HYTLASS(status)                                     \
  {                                                               \
    auto error = status;                                          \
    if (error != hytlass::Status::kSuccess) {                     \
      std::cerr << "HYTLASS error = "                              \
                << int(error) << " ("                             \
                << hytlassGetStatusString(error) << ")"           \
                << " at line " << __LINE__ << std::endl;          \
      std::abort();                                               \
    }                                                             \
  }

#define CHECK_HIP(func)                                                      \
  {                                                                           \
    hipError_t err = func;                                                   \
    if (err != hipSuccess) {                                                 \
      std::cerr << "[" << __FILE__ << ":" << __LINE__ << ", " << __FUNCTION__ \
                << "] "                                                       \
                << "HIP error(" << err << "), " << hipGetErrorString(err)   \
                << " when call " << #func << std::endl;                       \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
  }