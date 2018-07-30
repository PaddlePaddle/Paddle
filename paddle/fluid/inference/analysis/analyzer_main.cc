/*
 * This file implements analysizer -- an executation help to analyze and
 * optimize trained model.
 */
#include "paddle/fluid/inference/analysis/analyzer.h"
#include <gflags/gflags.h>
#include <glog/logging.h>

int main(int argc, char** argv) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  using paddle::inference::analysis::Analyzer;
  using paddle::inference::analysis::Argument;

  Argument argument;
  Analyzer analyzer;
  analyzer.Run(&argument);

  return 0;
}
