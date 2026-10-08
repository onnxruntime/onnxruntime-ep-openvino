// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <algorithm>
#include <cstring>
#include <ios>
#include <limits>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <utility>

namespace onnxruntime {
namespace openvino_ep_plugin {
namespace detail {

class LargeStringBuf final : public std::streambuf {
 public:
  explicit LargeStringBuf(std::string::size_type initial_capacity = 64 * 1024 * 1024) {
    storage_.reserve(std::min(initial_capacity, storage_.max_size()));
  }

  std::string TakeString() && {
    return std::move(storage_);
  }

 protected:
  std::streamsize xsputn(const char* data, std::streamsize count) override {
    if (count < 0) {
      throw std::invalid_argument("Negative stream write size.");
    }

    const auto write_size = static_cast<std::string::size_type>(count);
    if (position_ > storage_.max_size() || write_size > storage_.max_size() - position_) {
      throw std::length_error("Serialized data is too large.");
    }

    const auto end_position = position_ + write_size;
    if (end_position > storage_.size()) {
      EnsureCapacity(end_position);
      storage_.resize(end_position);
    }

    if (write_size != 0) {
      std::memcpy(storage_.data() + position_, data, write_size);
    }
    position_ = end_position;
    return count;
  }

  int_type overflow(int_type character = traits_type::eof()) override {
    if (traits_type::eq_int_type(character, traits_type::eof())) {
      return traits_type::not_eof(character);
    }

    const char value = traits_type::to_char_type(character);
    return xsputn(&value, 1) == 1 ? character : traits_type::eof();
  }

  pos_type seekoff(off_type offset,
                   std::ios_base::seekdir direction,
                   std::ios_base::openmode) override {
    const auto base = direction == std::ios_base::beg
                          ? int64_t{0}
                      : direction == std::ios_base::cur
                          ? static_cast<int64_t>(position_)
                      : direction == std::ios_base::end ? static_cast<int64_t>(storage_.size())
                                                        : -1;
    if (base < 0) {
      return pos_type(off_type(-1));
    }

    const auto offset_value = static_cast<int64_t>(offset);
    if ((offset_value > 0 && base > std::numeric_limits<int64_t>::max() - offset_value) ||
        (offset_value < 0 && base < std::numeric_limits<int64_t>::min() - offset_value)) {
      return pos_type(off_type(-1));
    }

    const auto target = base + offset_value;
    if (target < 0 || std::cmp_greater(target, storage_.max_size())) {
      return pos_type(off_type(-1));
    }

    position_ = static_cast<std::string::size_type>(target);
    return pos_type(position_);
  }

  pos_type seekpos(pos_type position, std::ios_base::openmode) override {
    if (position < pos_type(0)) {
      return pos_type(off_type(-1));
    }

    position_ = static_cast<std::string::size_type>(position);
    return position;
  }

 private:
  void EnsureCapacity(std::string::size_type required_size) {
    if (required_size <= storage_.capacity()) {
      return;
    }

    const auto current_capacity = storage_.capacity();
    const auto growth = current_capacity / 2;
    const auto max_size = storage_.max_size();
    const auto grown_capacity = growth > max_size - current_capacity ? max_size : current_capacity + growth;
    storage_.reserve(std::max(required_size, grown_capacity));
  }

  std::string storage_;
  std::string::size_type position_ = 0;
};

}  // namespace detail
}  // namespace openvino_ep_plugin
}  // namespace onnxruntime