#include "permutations_cxx.h"

#include <algorithm>
#include <unordered_map>

void Permutations(dictionary_t& dictionary) {
  std::unordered_map<std::string, std::vector<std::string>> groups;

  for (const auto& [word, _] : dictionary) {
    std::string key = word;
    std::sort(key.begin(), key.end());
    groups[key].push_back(word);
  }

  for (auto& [word, permutations] : dictionary) {
    std::string key = word;
    std::sort(key.begin(), key.end());

    permutations.clear();
    for (const auto& candidate : groups[key]) {
      if (candidate != word) {
        permutations.push_back(candidate);
      }
    }

    std::sort(permutations.begin(), permutations.end(),
              std::greater<std::string>());
  }
}
