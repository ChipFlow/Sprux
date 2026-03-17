/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
// Copyright (c) Robert Taylor, 2026. All rights reserved.
// Licensed under the MIT license found in the LICENSE file.

#pragma once

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <ostream>
#include <string>
#include <vector>

namespace Sprux {

struct BenchRecord {
  std::string problem;
  std::string solver;
  std::string operation;
  std::vector<double> times_sec;
  double median_sec;
};

// Escape a string for JSON output
inline std::string jsonEscape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        out += c;
    }
  }
  return out;
}

inline void writeJson(std::ostream& os, const std::vector<BenchRecord>& records) {
  os << "{\n";
  os << "  \"meta\": {\n";

  // Timestamp in ISO 8601
  auto now = std::chrono::system_clock::now();
  auto time_t_now = std::chrono::system_clock::to_time_t(now);
  struct tm tm_buf;
  gmtime_r(&time_t_now, &tm_buf);
  char timeBuf[64];
  strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
  os << "    \"timestamp\": \"" << timeBuf << "\"\n";

  os << "  },\n";
  os << "  \"results\": [\n";
  for (size_t i = 0; i < records.size(); i++) {
    const auto& r = records[i];
    os << "    {\n";
    os << "      \"problem\": \"" << jsonEscape(r.problem) << "\",\n";
    os << "      \"solver\": \"" << jsonEscape(r.solver) << "\",\n";
    os << "      \"operation\": \"" << jsonEscape(r.operation) << "\",\n";
    os << "      \"times_sec\": [";
    for (size_t j = 0; j < r.times_sec.size(); j++) {
      if (j > 0) os << ", ";
      os << std::fixed << std::setprecision(6) << r.times_sec[j];
    }
    os << "],\n";
    os << "      \"median_sec\": " << std::fixed << std::setprecision(6) << r.median_sec << "\n";
    os << "    }" << (i + 1 < records.size() ? "," : "") << "\n";
  }
  os << "  ]\n";
  os << "}\n";
}

// Compute median of a vector of doubles
inline double computeMedian(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

}  // namespace Sprux
