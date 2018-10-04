#pragma once
#include <set>
#include <vector>
#include <utility>
#include <stdint.h>
#include <cstddef>
#include <string>


// Character Error Rate (CER) and Word Error Rate (WER)
class Metrics
{
public:
  // CTOR: pass characters which can occur in words
  explicit Metrics(const std::set<uint32_t>& wordChars);
  
  // add result for sample result: pass ground truth text and recognized text
  void addResult(const std::string& gt, const std::string& rec);
  
  // get CER and WER of the (accumulated) text so far
  double getCER() const;
  double getWER() const;
  double getAccuracy() const;
  
private:
  std::set<uint32_t> m_wordChars;
  size_t editDistance(const std::string& t1, const std::string& t2);
  size_t m_numChars=0, m_edChars=0;
  size_t m_numWords = 0, m_edWords = 0;
  size_t m_numCorrectWords = 0, m_totalWordsSoFar = 0;

  std::pair<std::vector<uint32_t>, std::vector<uint32_t>> getWordIDStrings(const std::vector<uint32_t>& t1, const std::vector<uint32_t>& t2) const;
};


