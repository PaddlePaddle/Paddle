#pragma once
#include <vector>
#include <list>
#include <unordered_map>

namespace paddle {
namespace framework {

template <class K, class V>
class LRUCache {
public:
   LRUCache(int capacity) {
       this->capacity = capacity;
   }

   int Size() {
     std::unique_lock<std::mutex> lock(mutex_);
     return q.size();
   }

   V Get(K key) {
       std::unique_lock<std::mutex> lock(mutex_);
       return GetUnlock(key)
/*
       auto itr = m.find(key);
       if(itr == m.end()) {
           return -1;
       }
       auto pair = *(itr->second);
       q.erase(itr->second);
       m.erase(itr);
       q.push_front(pair);
       m[key] = q.begin();
       return pair.second;
*/
   }

   V GetUnlock(K key) {
       //std::unique_lock<std::mutex> lock(mutex_);
       auto itr = m.find(key);
       if(itr == m.end()) {
           return -1;
       }
       auto pair = *(itr->second);
       q.erase(itr->second);
       m.erase(itr);
       q.push_front(pair);
       m[key] = q.begin();
       return pair.second;
   }
   
   void Put(K key, V value) {
       std::unique_lock<std::mutex> lock(mutex_);
       auto itr = m.find(key);
       if(itr == m.end()) {
           if(m.size() == capacity) {
               auto pair = q.back();
               q.pop_back();
               m.erase(m.find(pair.first));
           }
           q.push_front(make_pair(key, value));
           m[key] = q.begin();
       } else {
           auto pair = *(itr->second);
           pair.second = value;
           q.erase(itr->second);
           q.push_front(pair);
           m[key] = q.begin();
       }
       rand_queue.push_back(key);
   }

   V GetRandom() {
     std::unique_lock<std::mutex> lock(mutex_);
     int size = rand_queue.size();
     for(int i = 0; i < size; ++i) {
     //while(!rand_queue.empty()) {
       K key = rand_queue.front();
       rand_queue.pop_front();
       if (m.find(key) == m.end()) {
         continue;
       }
       rand_queue.push_back(key);
       return GetUnlock(key);
     }
     return V();
   }

   // <key, value>
   std::list<std::pair<K, V>> q;
   typedef std::list<pair<K, V>>::iterator iter;
   std::unordered_map<K, iter> m;
   int capacity;
   std::queue<K> rand_queue;

  //std::deque<T> data_;
  //int capacity;
  //std::mutex_;

};

}
}
