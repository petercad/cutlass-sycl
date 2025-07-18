#include <sycl/sycl.hpp>

#include "cute/tensor.hpp"

using namespace sycl;
using namespace cute;

int main()
{
  queue Q;

  auto mem = malloc_device<half>(256, Q);

  Q.parallel_for(nd_range<1>(16, 16), [=](nd_item<1> id) [[sycl::reqd_sub_group_size(16)]] {
    auto x = make_tensor<half_t>(Shape<_8>{}, Stride<_1>{});

    XE_2D_U16x8x16_LD_N::copy(mem, 16, 16, 16, intel::coord_t{0, 0}, &*x.data());

    auto &xe = *reinterpret_cast<array<half_t, 8> *>(x.data());

    half_t y{2.0f};

    SGCoop<cute::plus> op{};
    xe = op(xe, SGUniform{y});

    XE_2D_U16x8x16_ST_N::copy(mem, 16, 16, 16, intel::coord_t{0, 0}, &*x.data());
  });
}

