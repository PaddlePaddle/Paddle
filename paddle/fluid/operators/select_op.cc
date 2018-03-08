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

#include <thread>
#include <vector>
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/channel.h"

namespace paddle {
namespace operators {

static constexpr char kCasesBlock[] = "sub_block";

static constexpr char kCaseToExecute[] = "case_to_execute";

static constexpr char kAttrCaseIndex[] = "case_index";
static constexpr char kAttrCaseType[] = "case_type";
static constexpr char kAttrCaseChannel[] = "case_channel";
static constexpr char kAttrCaseChannelVar[] = "case_channel_var";

class SelectOp : public framework::OperatorBase {
public:
  SelectOp(const std::string &type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
          : framework::OperatorBase(type, inputs, outputs, attrs) {}

private:
  enum SelectOpCaseType {
      DEFAULT = 0,
      SEND = 1,
      RECEIVE = 2
  };

  struct SelectOpCase {
      int caseIndex;
      SelectOpCaseType caseType;
      std::string channelName;
      std::string varName;

      SelectOpCase() {}

      SelectOpCase(int caseIndex, SelectOpCaseType caseType,
                   std::string channelName, std::string varName) :
              caseIndex(caseIndex), caseType(caseType),
              channelName(channelName), varName(varName) {}
  };

  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    framework::BlockDesc *casesBlock =
            Attr<framework::BlockDesc *>(kCasesBlock);

    framework::Scope &casesBlockScope = scope.NewScope();

    std::string caseToExecuteVarName = Input(kCaseToExecute);
    framework::Variable *caseToExecuteVar =
            casesBlockScope.FindVar(caseToExecuteVarName);

    // Construct cases from "conditional_block_op"(s) in the casesBlock
    std::vector<std::shared_ptr<SelectOpCase>> cases =
            ParseAndShuffleCases(casesBlock);

    // Get all unique channels involved in select
    std::set<framework::ChannelHolder *> channelsSet;
    for (auto c : cases) {
      if (!c->channelName.empty()) {
        auto channelVar = scope.FindVar(c->channelName);
        framework::ChannelHolder *ch =
                channelVar->GetMutable<framework::ChannelHolder>();

        if (channelsSet.find(ch) == channelsSet.end()) {
          channelsSet.insert(ch);
        }
      }
    }

    // Order all channels by their pointer address
    std::vector<framework::ChannelHolder *> channels(channelsSet.begin(),
                                                     channelsSet.end());
    std::sort(channels.begin(), channels.end());

    // Poll all cases
    int caseToExecute = pollCases(&scope, &cases, channels);

    // At this point, the case to execute has already been determined,
    // so we can proceed with executing the cases block
    framework::LoDTensor *caseToExecuteTensor =
            caseToExecuteVar->GetMutable<framework::LoDTensor>();
    caseToExecuteTensor->data<int>()[0] = caseToExecute;

    // Execute the cases block, only one case will be executed since we set the
    // case_to_execute value to the index of the case we want to execute
    framework::Executor executor(dev_place);
    framework::ProgramDesc *program = casesBlock->Program();
    executor.Run(*program, &casesBlockScope, casesBlock->ID(),
                 false /*create_local_scope*/);
  }

  /**
   * Goes through all operators in the "cases" block and processes
   * "conditional_block" operators.  These operators are mapped to our
   * SelectOpCase objects.  We randomize the case orders, and set the
   * default case (if any exists) as the last case)
   * @param casesBlock
   * @return
   */
  std::vector<std::shared_ptr<SelectOpCase>> ParseAndShuffleCases(
          framework::BlockDesc *casesBlock) const {
    std::vector<std::shared_ptr<SelectOpCase>> cases;
    std::shared_ptr<SelectOpCase> defaultCase;

    if (casesBlock != nullptr) {
      // TODO(thuan): Enforce cases block only has conditional_ops
      for (auto* caseOp : casesBlock->AllOps()) {
        if (caseOp->Type() != "conditional_block") {
          // We only want to process conditional_block_ops
          continue;
        }
        framework::Attribute caseIndexAttr = caseOp->GetAttr(kAttrCaseIndex);
        int caseIndex = boost::get<int>(caseIndexAttr);

        framework::Attribute caseTypeAttr = caseOp->GetAttr(kAttrCaseType);
        SelectOpCaseType caseType = static_cast<SelectOpCaseType>(
                boost::get<int>(caseTypeAttr));

        framework::Attribute caseChannelAttr =
                caseOp->GetAttr(kAttrCaseChannel);
        std::string caseChannel = boost::get<std::string>(caseChannelAttr);

        framework::Attribute caseChannelVarAttr =
                caseOp->GetAttr(kAttrCaseChannelVar);
        std::string caseChannelVar = boost::get<std::string>(
                caseChannelVarAttr);

        auto c = std::make_shared<SelectOpCase>(caseIndex, caseType,
                                                caseChannel, caseChannelVar);

        if (caseType == DEFAULT) {
          // TODO(thuan): Enforce default isn't already set
          defaultCase = c;
        } else {
          cases.push_back(c);
        }
      }
    }

    // Randomly sort cases, with default case being last
    std::random_shuffle(cases.begin(), cases.end());
    if (defaultCase != nullptr) {
      cases.push_back(defaultCase);
    }

    return cases;
  }

  /**
   * This method will recursively poll the cases and
   * @param scope
   * @param cases
   * @return
   */
  int pollCases(const framework::Scope *scope,
                std::vector<std::shared_ptr<SelectOpCase>> *cases,
                std::vector<framework::ChannelHolder *> channels) const {
    // Lock all involved channels
    lockChannels(channels);

    int caseToExecute = -1;

    std::vector<std::shared_ptr<SelectOpCase>>::iterator it = cases->begin();
    while (it != cases->end()) {
      std::shared_ptr<SelectOpCase> c = *it;
      std::cout << "CASE: " << c->caseType << std::endl;

//      auto chVar = scope->FindVar(c->channelName);
//      framework::ChannelHolder *ch =
//              chVar->GetMutable<framework::ChannelHolder>();
      // TODO(thuan): Check if operation is send to closed channel.  If it is,
      //    then PADDLE_ENFORCE
      // TODO(thuan): Check ch->CanSend() or ch->CanRecv().  If it is, then
      //    execute case body and exit

      if (c->caseType == DEFAULT) {
        // TODO(thuan): execute default case body and exit
        caseToExecute = c->caseIndex;
        break;
      }

      ++it;
    }

    if (caseToExecute == -1) {
      // None of the cases are eligible to execute, enqueue current thread
      // into all the sending/receiving Q of each involved channel
      // TODO(thuan): Atomically unlock all channels and sleep current thread
    }

    return caseToExecute;
  }

  void lockChannels(std::vector<framework::ChannelHolder *> chs) const {
    std::vector<framework::ChannelHolder *>::iterator it = chs.begin();
    while (it != chs.end()) {
      // framework::ChannelHolder * ch = *it;
      // TODO(thuan): Call ch.lock when its implemented;
      ++it;
    }
  }

  void unlockChannels(std::vector<framework::ChannelHolder *> chs) const {
    std::vector<framework::ChannelHolder *>::reverse_iterator it = chs.rbegin();
    while (it != chs.rend()) {
      // framework::ChannelHolder * ch = *it;
      // TODO(thuan): Call ch.unlock when its implemented;
      ++it;
    }
  }
};

class SelectOpMaker : public framework::OpProtoAndCheckerMaker {
  public:
    SelectOpMaker(OpProto *proto, OpAttrChecker *op_checker)
            : OpProtoAndCheckerMaker(proto, op_checker) {
      AddAttr<framework::BlockDesc *>(kCasesBlock,
                                      "The cases block inside select_op");
      AddComment(R"DOC(
)DOC");
    }
};

// TODO(thuan): Implement Gradient Operator for SELECT_OP

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(select, paddle::operators::SelectOp,
                  paddle::framework::EmptyGradOpMaker,
                  paddle::operators::SelectOpMaker);
