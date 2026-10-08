// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#include "env.h"
#include "path_utils.h"
#include <type_traits>

namespace ovep_tests {
namespace data_utils {
namespace fs = std::filesystem;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline bool is_power_of_2(const T number) {
  return ((number & (number - 1)) == 0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline T align(const T value, const size_t alignment) {
  assert(is_power_of_2(alignment));

  T mask = static_cast<T>(alignment) - 1;
  return (value + mask) & ~mask;
}

///////////////////////////////////////////////////////////////////////
inline size_t align(size_t n, size_t alignment) {
  return n + (alignment - 1) - ((n - 1) % alignment);
}

///////////////////////////////////////////////////////////////////////
template <std::integral T>
inline T ceil_div(T x, T y) {
  return (x + y - 1) / y;
}

///////////////////////////////////////////////////////////////////////
inline void seed_rand(unsigned int seed) {
  std::srand(seed);
}

///////////////////////////////////////////////////////////////////////
inline void seed_rand(void) {
  auto seed_str = get_environment_variable("OVEP_TESTS_RANDOM_SEED");
  auto seed = seed_str.has_value() ? static_cast<unsigned int>(std::stoul(*seed_str)) : 0xC0FFEEu;
  seed_rand(seed);
}

///////////////////////////////////////////////////////////////////////
inline float randf(float minv = 0.f, float maxv = 1.f) {
  if (minv >= maxv)
    return minv;

  return (float)(std::rand() / (float)RAND_MAX) * (maxv - minv) + minv;
}

///////////////////////////////////////////////////////////////////////
template <typename T>
inline T rand(T min, T max) {
  return std::rand() / (RAND_MAX / (max - min + 1) + 1) + min;
}

////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline void read_binary_file(const fs::path& path, std::vector<T>& out) {
  OVEP_TESTS_LOG_DEBUG << "ovep_tests::data_utils::read_binary_file( path: " << path_to_utf8(path) << " ): {";

  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  auto size = ifs.tellg();
  ifs.seekg(0, std::ios::beg);

  if (size > 0) {
    if (out.size() < static_cast<size_t>(size)) {
      out.resize(size);
    }

    ifs.read((char*)out.data(), size);
  }

  OVEP_TESTS_LOG_DEBUG << "...OK }";
}

////////////////////////////////////////////////////////////////////////////////////////
template <typename T>
inline std::vector<T> read_binary_file(const fs::path& path) {
  std::vector<T> out;

  read_binary_file<T>(path, out);

  return out;
}

////////////////////////////////////////////////////////////////////////////////////////
inline void write_binary_file(const fs::path& path, const std::vector<std::byte>& data) {
  std::ofstream fout(path, std::ios::out | std::ios::binary);
  fout.write((char*)data.data(), data.size());
}

///////////////////////////////////////////////////////////////////////
template <std::ranges::forward_range R1, std::ranges::forward_range R2>
inline double cosine_similarity(const R1& vec1, const R2& vec2) {
  double dot_product = 0.0;
  double norm_a = 0.0;
  double norm_b = 0.0;

  EXPECT_TRUE(std::distance(vec1.begin(), vec1.end()) == std::distance(vec2.begin(), vec2.end()));

  for (auto it1 = vec1.begin(), it2 = vec2.begin(); it1 != vec1.end(); ++it1, ++it2) {
    double vec1_double = static_cast<double>(*it1);
    double vec2_double = static_cast<double>(*it2);

    dot_product += vec1_double * vec2_double;
    norm_a += vec1_double * vec1_double;
    norm_b += vec2_double * vec2_double;
  }

  EXPECT_FALSE(norm_a == 0.0 || norm_b == 0.0);

  return dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

///////////////////////////////////////////////////////////////////////
template <std::ranges::forward_range R1, std::ranges::forward_range R2>
inline double l2_norm(const R1& vec1, const R2& vec2) {
  double sum_of_squares = 0.0;

  EXPECT_TRUE(std::distance(vec1.begin(), vec1.end()) == std::distance(vec2.begin(), vec2.end()));

  for (auto it1 = vec1.begin(), it2 = vec2.begin(); it1 != vec1.end(); ++it1, ++it2) {
    double diff = static_cast<double>(*it1) - static_cast<double>(*it2);
    sum_of_squares += diff * diff;
  }

  return std::sqrt(sum_of_squares);
}

///////////////////////////////////////////////////////////////////////
template <std::ranges::forward_range R1, std::ranges::forward_range R2>
inline double mean_absolute_error(const R1& vec1, const R2& vec2) {
  double mae = 0.0;

  auto size1 = std::distance(vec1.begin(), vec1.end());
  auto size2 = std::distance(vec2.begin(), vec2.end());

  if (size1 != size2 || size1 == 0) {
    return mae;
  }

  for (auto it1 = vec1.begin(), it2 = vec2.begin(); it1 != vec1.end(); ++it1, ++it2) {
    double a = static_cast<double>(*it1);
    double b = static_cast<double>(*it2);
    mae += std::abs(a - b);
  }
  return mae / size1;
}

///////////////////////////////////////////////////////////////////////
template <std::ranges::forward_range R1, std::ranges::forward_range R2>
inline double mean_squared_error(const R1& vec1, const R2& vec2) {
  double mse = 0.0;

  auto size1 = std::distance(vec1.begin(), vec1.end());
  auto size2 = std::distance(vec2.begin(), vec2.end());

  if (size1 != size2 || size1 == 0) {
    return mse;
  }

  for (auto it1 = vec1.begin(), it2 = vec2.begin(); it1 != vec1.end(); ++it1, ++it2) {
    double diff = static_cast<double>(*it1) - static_cast<double>(*it2);
    mse += diff * diff;
  }
  return mse / size1;
}

////////////////////////////////////////////////////////
template <std::ranges::forward_range R>
inline auto median(R& vec) {
  using elem_t = std::ranges::range_value_t<R>;
  std::vector<elem_t> sorted_vec(vec.begin(), vec.end());
  std::sort(sorted_vec.begin(), sorted_vec.end());

  auto mid_index = std::distance(vec.begin(), vec.end()) / 2;

  return sorted_vec.size() % 2 == 0
             ? static_cast<elem_t>((sorted_vec[mid_index - 1] + sorted_vec[mid_index]) / 2.0)
             : sorted_vec[mid_index];
}

////////////////////////////////////////////////////////
template <std::ranges::forward_range R>
inline auto min(R& vec) {
  using elem_t = std::ranges::range_value_t<R>;
  auto distance = std::distance(vec.begin(), vec.end());
  return distance > 0 ? *std::min_element(vec.begin(), vec.end()) : elem_t{};
}

////////////////////////////////////////////////////////
template <std::ranges::forward_range R>
inline auto max(R& vec) {
  using elem_t = std::ranges::range_value_t<R>;
  auto distance = std::distance(vec.begin(), vec.end());
  return distance > 0 ? *std::max_element(vec.begin(), vec.end()) : elem_t{};
}

////////////////////////////////////////////////////////
template <std::ranges::forward_range R>
inline auto average(R& vec) {
  using elem_t = std::ranges::range_value_t<R>;
  auto distance = std::distance(vec.begin(), vec.end());
  return distance > 0 ? static_cast<elem_t>(std::reduce(vec.begin(), vec.end()) / distance) : elem_t{};
}

}  // namespace data_utils
}  // namespace ovep_tests
