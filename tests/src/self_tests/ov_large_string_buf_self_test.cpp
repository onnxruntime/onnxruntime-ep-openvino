// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include <ostream>
#include <cstdlib>
#include <limits>
#include <string>

#include "gtest/gtest.h"
#include "ov_large_string_buf.h"

namespace onnxruntime::openvino_ep_plugin {

TEST(LargeStringBuf, GrowsAndSupportsSeekWrites) {
  detail::LargeStringBuf buffer(4);
  std::ostream stream(&buffer);

  stream.write("0123456789", 10);
  ASSERT_TRUE(stream.good());

  stream.seekp(3, std::ios_base::beg);
  ASSERT_TRUE(stream.good());
  stream.write("AB", 2);
  ASSERT_TRUE(stream.good());

  stream.seekp(0, std::ios_base::end);
  ASSERT_TRUE(stream.good());
  stream.put('!');
  ASSERT_TRUE(stream.good());

  EXPECT_EQ(std::move(buffer).TakeString(), "012AB56789!");
}

TEST(LargeStringBuf, RejectsOverflowingSeekOffsets) {
  detail::LargeStringBuf buffer;
  std::ostream stream(&buffer);

  stream.seekp(1, std::ios_base::beg);
  ASSERT_TRUE(stream.good());
  stream.seekp(std::numeric_limits<std::streamoff>::max(), std::ios_base::cur);

  EXPECT_FALSE(stream.good());
}

TEST(LargeStringBuf, SupportsMoreThanInt32MaxBytes) {
  detail::LargeStringBuf buffer;
  std::ostream stream(&buffer);
  const auto int32_max = static_cast<std::streamoff>(std::numeric_limits<int32_t>::max());

  stream.seekp(int32_max);
  ASSERT_TRUE(stream.good());
  stream.put('!');
  ASSERT_TRUE(stream.good());

  auto result = std::move(buffer).TakeString();
  ASSERT_EQ(result.size(), static_cast<size_t>(int32_max) + 1);
  EXPECT_EQ(result.back(), '!');
}

}  // namespace onnxruntime::openvino_ep_plugin