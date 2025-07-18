// TODO: license?

// Xe optimized array overloads for operator objects.

#pragma once

#include "cute/algorithm/functional.hpp"
#include "cute/util/sycl_vec.hpp"

namespace cute {

// SGCoop is a wrapper for subgroup-cooperative operations,
//    e.g. SGCoop<plus> -> subgroup-cooperative addition.
// The benefit vs. unwrapped operations is the ability to use SIMD32
//   arithmetic in hardware.
template <typename Op> struct SGCoop {};

// SGUniform is a wrapper for subgroup-uniform values (typically constants).
template <typename T> struct SGUniform {
private:
  T value_;
public:
  SGUniform() = default;
  constexpr SGUniform(T value) : value_(value) {}
  constexpr operator T() const { return value_; }
};


#define CUTE_SIMD_NM_BINARY_OP(op, simd, z, a, b) \
  asm( \
    #op " (M1_NM, " #simd ") %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<1;1,0>" \
    : "=rw"(z) : "rw"(a), "rw"(b) \
  );

#define CUTE_SIMD_NM_BINARY_OP_LUNIFORM(op, simd, z, a, b) \
  asm( \
    #op " (M1_NM, " #simd ") %0(0,0)<1> %1(0,0)<0;1,0> %2(0,0)<1;1,0>" \
    : "=rw"(z) : "rw.u"(a), "rw"(b) \
  );

#define CUTE_SIMD_NM_BINARY_OP_RUNIFORM(op, simd, z, a, b) \
  asm( \
    #op " (M1_NM, " #simd ") %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<0;1,0>" \
    : "=rw"(z) : "rw"(a), "rw.u"(b) \
  );

template <>
struct SGCoop<plus> {
  template <size_t N>
  array<half_t, N> operator()(array<half_t, N> const& lhs, array<half_t, N> const& rhs) const {
    array<half_t, N> result;
#if defined(SYCL_INTEL_TARGET) && defined(__SYCL_DEVICE_ONLY__)
    auto *result_ptr = reinterpret_cast<cute::intel::half2 *>(&result);
    auto *lhs_ptr    = reinterpret_cast<cute::intel::half2 const *>(&lhs);
    auto *rhs_ptr    = reinterpret_cast<cute::intel::half2 const *>(&rhs);

    CUTE_UNROLL
    for (size_t i = 0; i < N / 2; ++i) {
      CUTE_SIMD_NM_BINARY_OP(add, 32, result_ptr[i], lhs_ptr[i], rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      CUTE_SIMD_NM_BINARY_OP(add, 16, result[N - 1], lhs[N - 1], rhs[N - 1]);
    }
#else
    for (size_t i = 0; i < N; i++)
      result[i] = plus{}(lhs[i], rhs[i]);
#endif
    return result;
  }

  template <size_t N>
  array<half_t, N> operator()(half_t const& lhs, array<half_t, N> const& rhs) const {
    array<half_t, N> result;
#if defined(SYCL_INTEL_TARGET) && defined(__SYCL_DEVICE_ONLY__)
    auto *result_ptr = reinterpret_cast<cute::intel::half2 *>(&result);
    auto *rhs_ptr    = reinterpret_cast<cute::intel::half2 const *>(&rhs);
    cute::intel::half2 lhs2{_Float16(lhs), _Float16(lhs)};

    CUTE_UNROLL
    for (size_t i = 0; i < N / 2; ++i) {
      CUTE_SIMD_NM_BINARY_OP(add, 32, result_ptr[i], lhs2, rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      CUTE_SIMD_NM_BINARY_OP(add, 16, result[N - 1], _Float16(lhs), rhs[N - 1]);
    }
#else
    for (size_t i = 0; i < N; i++)
      result[i] = plus{}(lhs, rhs[i]);
#endif
    return result;
  }

  template <size_t N>
  array<half_t, N> operator()(SGUniform<half_t> const& lhs, array<half_t, N> const& rhs) const {
    array<half_t, N> result;
#if defined(SYCL_INTEL_TARGET) && defined(__SYCL_DEVICE_ONLY__)
    auto *result_ptr = reinterpret_cast<cute::intel::half2 *>(&result);
    auto *rhs_ptr    = reinterpret_cast<cute::intel::half2 const *>(&rhs);

    CUTE_UNROLL
    for (size_t i = 0; i < N / 2; ++i) {
      CUTE_SIMD_NM_BINARY_OP_LUNIFORM(add, 32, result_ptr[i], _Float16(half_t(lhs)), rhs_ptr[i]);
    }

    if constexpr (N % 2) {
      CUTE_SIMD_NM_BINARY_OP_LUNIFORM(add, 16, result[N - 1], _Float16(half_t(lhs)), rhs[N - 1]);
    }
#else
    for (size_t i = 0; i < N; i++)
      result[i] = plus{}(half_t(lhs), rhs[i]);
#endif
    return result;
  }


  template <size_t N>
  array<half_t, N> operator()(array<half_t, N> const& lhs, half_t const& rhs) const {
    array<half_t, N> result;
#if defined(SYCL_INTEL_TARGET) && defined(__SYCL_DEVICE_ONLY__)
    auto *result_ptr = reinterpret_cast<cute::intel::half2 *>(&result);
    auto *lhs_ptr    = reinterpret_cast<cute::intel::half2 const *>(&lhs);
    cute::intel::half2 rhs2{_Float16(rhs), _Float16(rhs)};

    CUTE_UNROLL
    for (size_t i = 0; i < N / 2; ++i) {
      CUTE_SIMD_NM_BINARY_OP(add, 32, result_ptr[i], lhs_ptr[i], rhs2);
    }

    if constexpr (N % 2) {
      CUTE_SIMD_NM_BINARY_OP(add, 16, result[N - 1], lhs[N - 1], _Float16(rhs));
    }
#else
    for (size_t i = 0; i < N; i++)
      result[i] = plus{}(lhs[i], rhs);
#endif
    return result;
  }

  template <size_t N>
  array<half_t, N> operator()(array<half_t, N> const& lhs, SGUniform<half_t> const& rhs) const {
    array<half_t, N> result;
#if defined(SYCL_INTEL_TARGET) && defined(__SYCL_DEVICE_ONLY__)
    auto *result_ptr = reinterpret_cast<cute::intel::half2 *>(&result);
    auto *lhs_ptr    = reinterpret_cast<cute::intel::half2 const *>(&lhs);

    CUTE_UNROLL
    for (size_t i = 0; i < N / 2; ++i) {
      CUTE_SIMD_NM_BINARY_OP_RUNIFORM(add, 32, result_ptr[i], lhs_ptr[i], _Float16(half_t(rhs)));
    }

    if constexpr (N % 2) {
      CUTE_SIMD_NM_BINARY_OP_RUNIFORM(add, 16, result[N - 1], lhs[N - 1], _Float16(half_t(rhs)));
    }
#else
    for (size_t i = 0; i < N; i++)
      result[i] = plus{}(lhs[i], half_t(rhs));
#endif
    return result;
  }
};

#undef CUTE_SIMD_NM_BINARY_OP
#undef CUTE_SIMD_NM_BINARY_OP_LUNIFORM
#undef CUTE_SIMD_NM_BINARY_OP_RUNIFORM

} /* namespace cute */