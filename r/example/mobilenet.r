#!/usr/bin/env Rscript

library(reticulate) # call Python library

use_python("/opt/python3.8/bin/python")

np <- import("numpy")
paddle <- import("paddle.base.core")

set_config <- function() {
    config <- paddle$AnalysisConfig("")
    config$set_model("data/model/__model__", "data/model/__params__")
    config$switch_use_feed_fetch_ops(FALSE)
    config$switch_specify_input_names(TRUE)
    config$enable_profile()

    return(config)
}

zero_copy_run_mobilenet <- function() {
    data <- np$loadtxt("data/data.txt")
    data <- data[0:(length(data) - 4)]
    result <- np$loadtxt("data/result.txt")
    result <- result[0:(length(result) - 4)]

    config <- set_config()
    predictor <- paddle$create_paddle_predictor(config)

    input_names <- predictor$get_input_names()
    input_tensor <- predictor$get_input_tensor(input_names[1])
    input_data <- np_array(data, dtype="float32")$reshape(as.integer(c(1, 3, 300, 300)))
    input_tensor$copy_from_cpu(input_data)

    predictor$zero_copy_run()

    output_names <- predictor$get_output_names()
    output_tensor <- predictor$get_output_tensor(output_names[1])
    output_data <- output_tensor$copy_to_cpu()
    output_data <- np_array(output_data)$reshape(as.integer(-1))
    #all.equal(output_data, result)
}

if (!interactive()) {
    zero_copy_run_mobilenet()
}
