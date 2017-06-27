#include "paddle/framework/scope.h"
#include "gtest/gtest.h"

TEST(Scope, Create) {
  using paddle::framework::Scope;
  using paddle::Error;
  using paddle::framework::Variable;
  using paddle::framework::AlreadyCreated;

  Scope* scope = new Scope();

  Error err = scope->CreateVariable("");
  EXPECT_FALSE(err.isOK());

  Variable* var1 = scope->GetVariable("a");
  EXPECT_EQ(var1, nullptr);

  Error err1 = scope->CreateVariable("a");
  EXPECT_TRUE(err1.isOK());

  Error err2 = scope->CreateVariable("a");
  EXPECT_EQ(err2, AlreadyCreated);

  Variable* var2 = scope->GetVariable("a");
  EXPECT_NE(var2, nullptr);

  Variable* var3 = scope->GetOrCreateVariable("b");
  EXPECT_NE(var3, nullptr);
}

TEST(Scope, Parent) {
  using paddle::framework::Scope;
  using paddle::framework::Variable;
  using paddle::Error;

  const auto parent_scope_ptr = std::shared_ptr<Scope>(new Scope());
  Scope* scope = new Scope(parent_scope_ptr);

  Error err = parent_scope_ptr->CreateVariable("a");
  EXPECT_TRUE(err.isOK());

  Variable* var1 = scope->GetVarLocally("a");
  EXPECT_EQ(var1, nullptr);

  Variable* var2 = scope->GetVariable("a");
  EXPECT_NE(var2, nullptr);
}