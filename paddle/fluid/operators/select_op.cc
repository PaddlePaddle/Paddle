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

#include <memory>
#include <thread>  // NOLINT
#include <vector>
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/concurrency/channel_util.h"

#include <boost/tokenizer.hpp>

namespace paddle {
namespace operators {

static constexpr char kX[] = "X";
static constexpr char kCaseToExecute[] = "case_to_execute";
static constexpr char kOutputs[] = "Out";

static constexpr char kCases[] = "cases";
static constexpr char kCasesBlock[] = "sub_block";

class SelectOp : public framework::OperatorBase {
 public:
  SelectOp(const std::string &type, const framework::VariableNameMap &inputs,
           const framework::VariableNameMap &outputs,
           const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  enum class SelectOpCaseType {
    DEFAULT = 0,
    SEND = 1,
    RECEIVE = 2,
  };

  struct SelectOpCase {
    int caseIndex;
    SelectOpCaseType caseType;
    std::string channelName;
    std::string varName;

    SelectOpCase() {}

    SelectOpCase(int caseIndex, SelectOpCaseType caseType,
                 std::string channelName, std::string varName)
        : caseIndex(caseIndex),
          caseType(caseType),
          channelName(channelName),
          varName(varName) {}
  };

  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    std::vector<std::string> casesConfigs =
        Attr<std::vector<std::string>>(kCases);

    framework::BlockDesc *casesBlock =
        Attr<framework::BlockDesc *>(kCasesBlock);

    framework::Scope &casesBlockScope = scope.NewScope();

    std::string caseToExecuteVarName = Input(kCaseToExecute);
    framework::Variable *caseToExecuteVar =
        casesBlockScope.FindVar(caseToExecuteVarName);

    // Construct cases from "conditional_block_op"(s) in the casesBlock
    std::vector<std::shared_ptr<SelectOpCase>> cases =
        ParseAndShuffleCases(&casesConfigs);

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
    int32_t caseToExecute = pollCases(&scope, &cases, channels);

    // At this point, the case to execute has already been determined,
    // so we can proceed with executing the cases block
    framework::LoDTensor *caseToExecuteTensor =
        caseToExecuteVar->GetMutable<framework::LoDTensor>();
    caseToExecuteTensor->data<int32_t>()[0] = caseToExecute;

    // Execute the cases block, only one case will be executed since we set the
    // case_to_execute value to the index of the case we want to execute
    framework::Executor executor(dev_place);
    framework::ProgramDesc *program = casesBlock->Program();
    executor.Run(*program, &casesBlockScope, casesBlock->ID(),
                 false /*create_local_scope*/);
  }

  /**
   * Goes through all operators in the casesConfigs and processes
   * "conditional_block" operators.  These operators are mapped to our
   * SelectOpCase objects.  We randomize the case orders, and set the
   * default case (if any exists) as the last case)
   * @param casesBlock
   * @return
   */
  std::vector<std::shared_ptr<SelectOpCase>> ParseAndShuffleCases(
      std::vector<std::string> *casesConfigs) const {
    std::vector<std::shared_ptr<SelectOpCase>> cases;
    std::shared_ptr<SelectOpCase> defaultCase;

    if (casesConfigs != nullptr) {
      boost::char_delimiters_separator<char> sep(false, ",", "");
      for (std::vector<std::string>::iterator itr = casesConfigs->begin();
           itr < casesConfigs->end(); ++itr) {
        std::string caseConfig = *itr;
        boost::tokenizer<> tokens(caseConfig, sep);

        boost::tokenizer<>::iterator tok_iter = tokens.begin();
        PADDLE_ENFORCE(tok_iter != tokens.end(), "Cannot get case index");
        std::string caseIndexString = *tok_iter;
        int caseIndex = std::stoi(caseIndexString);

        ++tok_iter;
        PADDLE_ENFORCE(tok_iter != tokens.end(), "Cannot get case type");
        std::string caseTypeString = *tok_iter;
        SelectOpCaseType caseType = (SelectOpCaseType)std::stoi(caseTypeString);

        std::string caseChannel;
        std::string caseChannelVar;

        ++tok_iter;
        if (caseType != SelectOpCaseType::DEFAULT) {
          PADDLE_ENFORCE(tok_iter != tokens.end(), "Cannot get case channel");
          caseChannel = *tok_iter;

          ++tok_iter;
          PADDLE_ENFORCE(tok_iter != tokens.end(),
                         "Cannot get case channel variable");
          caseChannelVar = *tok_iter;
        }

        auto c = std::make_shared<SelectOpCase>(caseIndex, caseType,
                                                caseChannel, caseChannelVar);

        if (caseType == SelectOpCaseType::DEFAULT) {
          PADDLE_ENFORCE(defaultCase == nullptr,
                         "Select can only contain one default case.");
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
   * This method will recursively poll the cases and determines if any case
   * condition is true.
   * If none of the cases conditions are true (and there is no default case),
   * then block
   * the thread.  The thread may be woken up by a channel operation, at which
   * point we
   * execute the case.
   * @param scope
   * @param cases
   * @param channels
   * @return
   */
  int32_t pollCases(const framework::Scope *scope,
                    std::vector<std::shared_ptr<SelectOpCase>> *cases,
                    std::vector<framework::ChannelHolder *> channels) const {
    // Lock all involved channels
    lockChannels(channels);

    std::atomic<int> caseToExecute(-1);

    std::vector<std::shared_ptr<SelectOpCase>>::iterator it = cases->begin();
    while (it != cases->end()) {
      std::shared_ptr<SelectOpCase> c = *it;

      auto chVar = scope->FindVar(c->channelName);
      framework::ChannelHolder *ch =
          chVar->GetMutable<framework::ChannelHolder>();

      switch (c->caseType) {
        case SelectOpCaseType::SEND:
          PADDLE_ENFORCE(!ch->IsClosed(), "Cannot send to a closed channel");
          if (ch->CanSend()) {
            // We can send to channel directly, send the data to channel
            // and execute case
            auto chVar = scope->FindVar(c->varName);
            concurrency::ChannelSend(ch, chVar);
            caseToExecute = c->caseIndex;
          }
          break;
        case SelectOpCaseType::RECEIVE:
          if (ch->CanReceive()) {
            // We can receive from channel directly, send the data to channel
            // and execute case
            auto chVar = scope->FindVar(c->varName);
            concurrency::ChannelReceive(ch, chVar);
            caseToExecute = c->caseIndex;
          }
          break;
        case SelectOpCaseType::DEFAULT:
          caseToExecute = c->caseIndex;
          break;
      }

      if (caseToExecute != -1) {
        // We found a case to execute, stop looking at other case statements
        break;
      }

      ++it;
    }

    if (caseToExecute == -1) {
      // None of the cases are eligible to execute, enqueue current thread
      // into all the sending/receiving queue of each involved channel
      std::atomic<bool> completed(false);
      std::recursive_mutex mutex;
      std::unique_lock<std::recursive_mutex> lock{mutex};
      // std::condition_variable_any selectCond;
      auto selectCond = std::make_shared<std::condition_variable_any>();

      std::recursive_mutex callbackMutex;
      pushThreadOnChannelQueues(scope, cases, selectCond, &caseToExecute,
                                &completed, &callbackMutex);

      // TODO(thuan): Atomically unlock all channels and sleep current thread
      unlockChannels(channels);
      selectCond->wait(lock, [&completed]() { return completed.load(); });

      // Select has been woken up by case operation
      lockChannels(channels);
      removeThreadOnChannelQueues(scope, cases);

      if (caseToExecute == -1) {
        // Recursively poll cases, since we were woken up by a channel close
        // TODO(thuan): Need to test if this is a valid case
        unlockChannels(channels);
        return pollCases(scope, cases, channels);
      }
    }

    // At this point, caseToExecute != -1, and we can proceed with executing
    // the case block
    unlockChannels(channels);

    return caseToExecute;
  }

  void lockChannels(std::vector<framework::ChannelHolder *> chs) const {
    std::vector<framework::ChannelHolder *>::iterator it = chs.begin();
    while (it != chs.end()) {
      framework::ChannelHolder *ch = *it;
      ch->Lock();
      ++it;
    }
  }

  void unlockChannels(std::vector<framework::ChannelHolder *> chs) const {
    std::vector<framework::ChannelHolder *>::reverse_iterator it = chs.rbegin();
    while (it != chs.rend()) {
      framework::ChannelHolder *ch = *it;
      ch->Unlock();
      ++it;
    }
  }

  void pushThreadOnChannelQueues(
      const framework::Scope *scope,
      std::vector<std::shared_ptr<SelectOpCase>> *cases,
      std::shared_ptr<std::condition_variable_any> rCond,
      std::atomic<int> *caseToExecute, std::atomic<bool> *completed,
      std::recursive_mutex *callbackMutex) const {
    std::vector<std::shared_ptr<SelectOpCase>>::iterator it = cases->begin();
    while (it != cases->end()) {
      std::shared_ptr<SelectOpCase> c = *it;

      auto chVar = scope->FindVar(c->channelName);
      framework::ChannelHolder *ch =
          chVar->GetMutable<framework::ChannelHolder>();

      std::function<bool(framework::ChannelAction channelAction)> cb =
          [&caseToExecute, &completed, &callbackMutex,
           c](framework::ChannelAction channelAction) {
            std::lock_guard<std::recursive_mutex> lock{*callbackMutex};

            bool canProcess = false;
            if (!(*completed)) {
              // If the channel wasn't closed, we set the caseToExecute index
              // as this current case
              if (channelAction != framework::ChannelAction::CLOSE) {
                *caseToExecute = c->caseIndex;
              }
              // This will allow our conditional variable to break out of wait
              *completed = true;
              canProcess = true;
            }

            return canProcess;
          };

      switch (c->caseType) {
        case SelectOpCaseType::SEND: {
          auto chOutputVar = scope->FindVar(c->varName);
          concurrency::ChannelAddToSendQ(ch, this, chOutputVar, rCond, cb);
          break;
        }
        case SelectOpCaseType::RECEIVE: {
          auto chOutputVar = scope->FindVar(c->varName);
          concurrency::ChannelAddToReceiveQ(ch, this, chOutputVar, rCond, cb);
          break;
        }
        default:
          break;
      }
      ++it;
    }
  }

  void removeThreadOnChannelQueues(
      const framework::Scope *scope,
      std::vector<std::shared_ptr<SelectOpCase>> *cases) const {
    std::vector<std::shared_ptr<SelectOpCase>>::iterator it = cases->begin();
    while (it != cases->end()) {
      std::shared_ptr<SelectOpCase> c = *it;

      auto chVar = scope->FindVar(c->channelName);
      framework::ChannelHolder *ch =
          chVar->GetMutable<framework::ChannelHolder>();
      switch (c->caseType) {
        case SelectOpCaseType::SEND: {
          ch->RemoveFromSendQ(this);
          break;
        }
        case SelectOpCaseType::RECEIVE: {
          ch->RemoveFromReceiveQ(this);
          break;
        }
        default:
          break;
      }
      ++it;
    }
  }
};

class SelectOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(kX,
             "A set of variables, which are required by operators inside the "
             "cases of Select Op")
        .AsDuplicable();
    AddInput(kCaseToExecute,
             "(Int) The variable the sets the index of the case to execute, "
             "after evaluating the channels being sent to and received from")
        .AsDuplicable();
    AddOutput(kOutputs,
              "A set of variables, which will be assigned with values "
              "generated by the operators inside the cases of Select Op.")
        .AsDuplicable();
    AddAttr<std::vector<std::string>>(kCases,
                                      "(String vector) Serialized list of"
                                      "all cases in the select op. Each"
                                      "case is serialized as: "
                                      "'<index>,<type>,<channel>,<value>'"
                                      "where type is 0 for default, 1 for"
                                      "send, and 2 for receive"
                                      "No channel and values are needed for"
                                      "default cases.");
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
