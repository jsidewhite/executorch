/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <tuple>

#include <executorch/kernels/portable/cpu/util/index_util.h>
#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using ScalarType = exec_aten::ScalarType;
using SizesType = exec_aten::SizesType;
using Tensor = exec_aten::Tensor;

std::tuple<Tensor&, Tensor&> min_out(
    RuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    bool keepdim,
    Tensor& min,
    Tensor& min_indices) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_min_max_args(in, dim, keepdim, min, min_indices),
      InvalidArgument,
      (std::tuple<Tensor&, Tensor&>({min, min_indices})));

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim, keepdim, min) == Error::Ok,
      InvalidArgument,
      (std::tuple<Tensor&, Tensor&>({min, min_indices})));

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(min_indices, min.sizes()) == Error::Ok,
      InvalidArgument,
      (std::tuple<Tensor&, Tensor&>({min, min_indices})));

  dim = dim < 0 ? dim + in.dim() : dim;

  return {min, min_indices};
}

} // namespace native
} // namespace executor
} // namespace torch
