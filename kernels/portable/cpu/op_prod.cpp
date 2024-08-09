/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/reduce_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using ScalarType = exec_aten::ScalarType;

Tensor& prod_out(
    RuntimeContext& ctx,
    const Tensor& in,
    optional<ScalarType> dtype,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, check_prod_out_args(in, dtype, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, {}) == Error::Ok, InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();
  constexpr auto name = "prod.int_out";


  return out;
}

Tensor& prod_int_out(
    RuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    bool keepdim,
    optional<ScalarType> dtype,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_reduction_args_single_dim(
          in, dim, keepdim, dtype, out, /*allow_empty_dim=*/true),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim, keepdim, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();
  constexpr auto name = "prod.int_out";

  

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
