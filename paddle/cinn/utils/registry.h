/**
 * This file is copied from dmlc-core project, all the rights are resolved by
 * original project. Following are the original header comment. Copyright (c)
 * 2015 by Contributors
 * @file registry.h
 * \brief Registry utility that helps to build registry singletons.
 */
#pragma once

#include <map>
#include <string>
#include <vector>

/**
 * \brief Registry class.
 *  Registry can be used to register global singletons.
 *  The most commonly use case are factory functions.
 *
 * @tparam EntryType Type of Registry entries,
 *     EntryType need to name a name field.
 */
template <typename EntryType>
class Registry {
 public:
  /** @return list of entries in the registry(excluding alias) */
  inline const std::vector<const EntryType *> &List() { return const_list_; }

  /** @return list all names registered in the registry, including alias */
  inline std::vector<std::string> ListAllNames() {
    std::vector<std::string> names;
    for (auto p = fmap_.begin(); p != fmap_.end(); ++p) {
      names.push_back(p->first);
    }
    return names;
  }
  /**
   * \brief Find the entry with corresponding name.
   * @param name name of the function
   * @return the corresponding function, can be NULL
   */
  inline const EntryType *Find(const std::string &name) {
    typename std::map<std::string, EntryType *>::const_iterator p =
        fmap_.find(name);
    if (p != fmap_.end()) {
      return p->second;
    } else {
      return nullptr;
    }
  }
  /**
   * \brief Add alias to the key_name
   * @param key_name The original entry key
   * @param alias The alias key.
   */
  /*   inline void AddAlias(const std::string &key_name, const std::string
    &alias) { EntryType *e = fmap_.at(key_name); if (fmap_.count(alias)) {
        CHECK_EQ(e, fmap_.at(alias)) << "Trying to register alias " << alias <<
    " for key " << key_name << " but "
                                     << alias << " is already taken";
      } else {
        fmap_[alias] = e;
      }
    } */
  /**
   * \brief Internal function to register a name function under name.
   * @param name name of the function
   * @return ref to the registered entry, used to set properties
   */
  inline EntryType &__REGISTER__(const std::string &name) {
    std::lock_guard<std::mutex> guard(registering_mutex);
    if (fmap_.count(name)) {
      return *fmap_[name];
    }

    EntryType *e = new EntryType();
    e->name = name;
    fmap_[name] = e;
    const_list_.push_back(e);
    entry_list_.push_back(e);
    return *e;
  }

  /**
   * \brief Internal function to either register or get registered entry
   * @param name name of the function
   * @return ref to the registered entry, used to set properties
   */
  inline EntryType &__REGISTER_OR_GET__(const std::string &name) {
    // Here if we use VLOG, we will get Seg Fault. Todo: Add VLOG and fix this.
    // @Haoze
    if (!fmap_.count(name)) {
      return __REGISTER__(name);
    } else {
      return *fmap_.at(name);
    }
  }

  /**
   * \brief get a singleton of the Registry.
   *  This function can be defined by CINN_REGISTRY_ENABLE.
   * @return get a singleton
   */
  static Registry *Global() {
    static Registry<EntryType> inst;
    return &inst;
  }

  Registry() = default;
  ~Registry() {
    for (size_t i = 0; i < entry_list_.size(); ++i) {
      delete entry_list_[i];
    }
  }

 protected:
  /** \brief list of entry types */
  std::vector<EntryType *> entry_list_;
  /** \brief list of entry types */
  std::vector<const EntryType *> const_list_;
  /** \brief map of name->function */
  std::map<std::string, EntryType *> fmap_;
  /** \brief lock guarding the registering*/
  std::mutex registering_mutex;
  /** \brief constructor */
  /** \brief destructor */
};

/**
 * \brief Common base class for function registry.
 *
 * \code
 *  // This example demonstrates how to use Registry to create a factory of
 * trees. struct TreeFactory : public FunctionRegEntryBase<TreeFactory,
 * std::function<Tree*()> > {
 *  };
 *
 *  // in a independent cc file
 *  namespace cinn {
 *  CINN_REGISTRY_ENABLE(TreeFactory);
 *  }
 *  // register binary tree constructor into the registry.
 *  CINN_REGISTRY_REGISTER(TreeFactory, TreeFactory, BinaryTree)
 *      .describe("Constructor of BinaryTree")
 *      .set_body([]() { return new BinaryTree(); });
 * \endcode
 *
 * @tparam EntryType The type of subclass that inheritate the base.
 * @tparam FunctionType The function type this registry is registered.
 */
template <typename EntryType, typename FunctionType>
class FunctionRegEntryBase {
 public:
  /** \brief name of the entry */
  std::string name;
  /** \brief description of the entry */
  std::string description;
  /** \brief additional arguments to the factory function */
  // std::vector<ParamFieldInfo> arguments;
  /** \brief Function body to create ProductType */
  FunctionType body;
  /** \brief Return type of the function */
  std::string return_type;

  /**
   * \brief Set the function body.
   * @param body Function body to set.
   * @return reference to self.
   */
  inline EntryType &set_body(FunctionType body) {
    this->body = body;
    return this->self();
  }
  /**
   * \brief Describe the function.
   * @param description The description of the factory function.
   * @return reference to self.
   */
  inline EntryType &describe(const std::string &description) {
    this->description = description;
    return this->self();
  }
  /**
   * \brief Set the return type.
   * @param type Return type of the function, could be Symbol or Symbol[]
   * @return reference to self.
   */
  inline EntryType &set_return_type(const std::string &type) {
    return_type = type;
    return this->self();
  }

 protected:
  /**
   * @return reference of self as derived type
   */
  inline EntryType &self() { return *(static_cast<EntryType *>(this)); }
};

/**
 * \brief Generic macro to register an EntryType
 *  There is a complete example in FactoryRegistryEntryBase.
 *
 * @param EntryType The type of registry entry.
 * @param EntryTypeName The typename of EntryType, must do not contain namespace
 * :: .
 * @param Name The name to be registered.
 * @sa FactoryRegistryEntryBase
 */
#define CINN_REGISTRY_REGISTER(EntryType, EntryTypeName, Name) \
  static EntryType &__make_##EntryTypeName##_##Name##__ =      \
      ::Registry<EntryType>::Global()->__REGISTER__(#Name)

#define CINN_STR_CONCAT_(__x, __y) __x##__y
#define CINN_STR_CONCAT(__x, __y) CINN_STR_CONCAT_(__x, __y)
