/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <iterator>
#include <sstream>
#include <vector>
#include "PaddleAPI.h"
#include "paddle/gserver/gradientmachines/GradientMachine.h"
#include "paddle/parameter/Argument.h"
#include "paddle/utils/Flags.h"

// used to represent partial sequence
struct Path {
  std::vector<int> ids;
  float logProb;
  paddle::MachineState machineState;

  Path() { logProb = 0; }

  Path(std::vector<int>& ids, float logProb, paddle::MachineState& machineState)
      : ids(ids), logProb(logProb), machineState(machineState) {}

  bool operator<(const Path& other) const { return (logProb > other.logProb); }
};

// Return top k (k == beam_size) optimal paths using beam search. The last
// element of inArgs is the Argument of feedback. gradMachine has MaxIdLayer
// as output and outArgs thus stores top k labels and their probabilities per
// position
static void findNBest(paddle::GradientMachine* gradMachine,
                      std::vector<paddle::Argument>& inArgs,
                      std::vector<Path>& finalPaths,
                      size_t bos_id,
                      size_t eos_id,
                      size_t max_length) {
  std::vector<Path> paths;
  Path emptyPath;
  paths.push_back(emptyPath);
  finalPaths.clear();
  gradMachine->resetState();
  paddle::Argument feedback = inArgs.back();
  feedback.ids->setElement(0, (int)(bos_id));
  float minFinalPathLogProb = 0;
  size_t beam = 0;
  int id;
  std::vector<paddle::Argument> outArgs;
  while (true) {  // iterate over each generated word
    std::vector<Path> newPaths;
    paddle::MachineState machineState;
    for (size_t j = 0; j < paths.size(); j++) {
      Path& path = paths[j];
      if (path.machineState.size() > 0) {
        gradMachine->setState(path.machineState);
        feedback.ids->setElement(0, path.ids.back());
      }
      gradMachine->forward(inArgs, &outArgs, paddle::PASS_TEST);
      gradMachine->getState(machineState);
      beam = outArgs[0].ids->getSize();
      for (size_t k = 0; k < beam; k++) {
        id = outArgs[0].ids->getElement(k);
        float prob = outArgs[0].in->getElement(0, k);
        std::vector<int> nids(path.ids);
        nids.push_back(id);
        float newLogProb = path.logProb + log(prob);
        Path newPath(nids, newLogProb, machineState);
        if (id == (int)eos_id || nids.size() >= max_length) {
          finalPaths.push_back(newPath);
          if (minFinalPathLogProb > newPath.logProb) {
            minFinalPathLogProb = newPath.logProb;
          }
        } else {
          newPaths.push_back(newPath);
        }
      }
    }

    if (newPaths.size() == 0) {
      break;
    }
    std::nth_element(newPaths.begin(),
                     newPaths.begin() + std::min(beam, newPaths.size()),
                     newPaths.end());
    if (newPaths.size() > beam) {
      newPaths.resize(beam);
    }
    // pathA < pathB means pathA.logProb > pathB.logProb
    float maxPathLogProb =
        std::min_element(newPaths.begin(), newPaths.end())->logProb;
    if (finalPaths.size() >= beam && minFinalPathLogProb >= maxPathLogProb) {
      break;
    }
    paths = newPaths;
  }  // end while

  std::partial_sort(finalPaths.begin(),
                    finalPaths.begin() + std::min(beam, finalPaths.size()),
                    finalPaths.end());
  if (finalPaths.size() > beam) {
    finalPaths.resize(beam);
  }
}

struct SequenceGeneratorPrivate {
  std::shared_ptr<paddle::GradientMachine> machine;
  std::shared_ptr<std::vector<std::string>> dict;
  size_t beginPos;
  size_t endPos;
  size_t maxLength;

  paddle::Argument feedback;

  template <typename T>
  inline T& cast(void* ptr) {
    return *(T*)(ptr);
  }

  inline void findNBest(std::vector<paddle::Argument>& inArgs,
                        std::vector<Path>& path) {
    ::findNBest(machine.get(), inArgs, path, beginPos, endPos, maxLength);
  }

  SequenceGeneratorPrivate()
      : dict(std::make_shared<std::vector<std::string>>()),
        beginPos(0UL),
        endPos(0UL),
        maxLength(0UL),
        feedback(__create_feedback__()) {}

 private:
  static paddle::Argument __create_feedback__() {
    paddle::Argument feedback;
    feedback.ids = paddle::IVector::create(/* size= */ 1, FLAGS_use_gpu);

    feedback.sequenceStartPositions =
        paddle::ICpuGpuVector::create(/* size= */ 2, /* useGpu= */ false);
    feedback.sequenceStartPositions->getMutableData(false)[0] = 0;
    feedback.sequenceStartPositions->getMutableData(false)[1] = 1;
    return feedback;
  }
};

SequenceGenerator::SequenceGenerator() : m(new SequenceGeneratorPrivate()) {}

SequenceGenerator::~SequenceGenerator() { delete m; }

class PathSequenceResults : public ISequenceResults {
  // ISequenceResults interface
 public:
  PathSequenceResults(const std::shared_ptr<std::vector<Path>>& path,
                      const std::shared_ptr<std::vector<std::string>>& dict)
      : path_(path), dict_(dict) {}

  size_t getSize() const { return path_->size(); }
  std::string getSentence(size_t id, bool split) const throw(RangeError) {
    if (id < getSize()) {
      Path& p = (*path_)[id];
      std::ostringstream sout;
      std::transform(p.ids.begin(),
                     p.ids.end(),
                     std::ostream_iterator<std::string>(sout, split ? " " : ""),
                     [&](int id) { return (*dict_)[id]; });
      return sout.str();
    } else {
      RangeError e;
      throw e;
    }
  }
  std::vector<int> getSequence(size_t id) const throw(RangeError) {
    if (id < getSize()) {
      Path& p = (*path_)[id];
      return p.ids;
    } else {
      RangeError e;
      throw e;
    }
  }
  float getScore(size_t id) const throw(RangeError) {
    if (id < getSize()) {
      Path& p = (*path_)[id];
      return p.logProb;
    } else {
      RangeError e;
      throw e;
    }
  }

 private:
  std::shared_ptr<std::vector<Path>> path_;
  std::shared_ptr<std::vector<std::string>> dict_;
};

ISequenceResults* SequenceGenerator::generateSequence(
    const Arguments& inArgs) const {
  auto& in_args =
      m->cast<std::vector<paddle::Argument>>(inArgs.getInternalArgumentsPtr());
  for (auto& arg : in_args) {
    arg.sequenceStartPositions = m->feedback.sequenceStartPositions;
  }
  in_args.push_back(m->feedback);
  auto path = std::make_shared<std::vector<Path>>();
  m->findNBest(in_args, *path);
  return new PathSequenceResults(path, m->dict);
}

SequenceGenerator* SequenceGenerator::createByGradientMachineSharedPtr(
    void* ptr) {
  SequenceGenerator* r = new SequenceGenerator();
  r->m->machine = r->m->cast<std::shared_ptr<paddle::GradientMachine>>(ptr);
  return r;
}

void SequenceGenerator::setDict(const std::vector<std::string>& dict) {
  *m->dict = dict;
}

void SequenceGenerator::setBos(size_t bos) { m->beginPos = bos; }

void SequenceGenerator::setEos(size_t eos) { m->endPos = eos; }

void SequenceGenerator::setMaxLength(size_t maxLength) {
  m->maxLength = maxLength;
}

void SequenceGenerator::setBeamSize(size_t beamSize) {
  if (beamSize != -1UL) {
    FLAGS_beam_size = beamSize;
  }
}

ISequenceResults::~ISequenceResults() {}
