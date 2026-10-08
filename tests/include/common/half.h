// Copyright (C) 2026 Intel Corporation
//
// SPDX-License-Identifier: MIT

#pragma once

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4201)  // nonstandard extension: nameless struct/union
#endif

struct half {
  struct s_fp32 {
    union {
      struct
      {
        unsigned int _mantis : 23;
        unsigned int _exp : 8;
        unsigned int _sign : 1;
      };
      uint32_t u;
      float f;
    };
  };

  union {
    struct
    {
      unsigned short _mantis : 10;
      unsigned short _exp : 5;
      unsigned short _sign : 1;
    };
    uint16_t fp16;
  };

  half() = default;
  half(float f) {
    _mantis = _exp = _sign = 0;

    if (f == 0) {
      return;
    }

    half& out = *this;
    s_fp32 in;
    in.f = f;

    out._sign = in._sign;
    if (in._exp == 0xFF) {
      // INF or NaN
      out._exp = 0x1F;
      out._mantis = (in._mantis >> 13) & 0x2FF;
    } else if (in._exp < 113) {
      // fp16 bias:15, fp32 bias:127
      // denormalized value
      out._exp = 0;
      out._mantis = 0;
    } else {
      out._exp = in._exp + 15 - 127;
      out._mantis = in._mantis >> 13;

      if ((in._mantis >> 12) & 1) {
        if (out._mantis == 0x3FF) {
          out._mantis = 0;
          out._exp += 1;
        } else {
          out._mantis += 1;
        }
      }
    }
  }
  half operator+(half rhs) { return half{((float)*this) + (float)rhs}; }
  half operator+=(half rhs) { return *this = half{((float)*this) + (float)rhs}; }
  half operator-(half rhs) {
    rhs._sign ^= 1;
    return *this + rhs;
  }
  half operator*(half rhs) { return half{((float)*this) * (float)rhs}; }
  bool operator>(half rhs) { return (float)*this > (float)rhs; }
  friend std::ostream& operator<<(std::ostream& os, half data);

  operator float() const {
    if (this->fp16 == 0)
      return 0;

    const half& in = *this;
    s_fp32 out;

    if (in._exp == 0x1F) {
      // infinite or NaN
      out._exp = 0xFF;  // 8bit
      out._mantis = in._mantis << 13;
    } else if (in._exp == 0) {
      // denormalized value
      out._exp = 0;
      out._mantis = 0;
    } else {
      out._exp = 127 - 15 + in._exp;
      out._mantis = in._mantis << 13;
    }
    out._sign = in._sign;

    return out.f;
  }
};

inline std::ostream& operator<<(std::ostream& os, half data) {
  os << (float)data;
  return os;
}

#ifdef _MSC_VER
#pragma warning(pop)
#endif
