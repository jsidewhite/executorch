/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/util/broadcast_util.h>
#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& where_out(
    RuntimeContext& ctx,
    const Tensor& cond,
    const Tensor& a,
    const Tensor& b,
    Tensor& out) {
  ScalarType cond_type = cond.scalar_type();
  ScalarType a_type = a.scalar_type();
  ScalarType b_type = b.scalar_type();
  ScalarType common_type = promoteTypes(a_type, b_type);
  ScalarType out_type = out.scalar_type();

  ET_KERNEL_CHECK(ctx, common_type == out_type, InvalidArgument, out);

  // Determine output size and resize for dynamic shapes
  ET_KERNEL_CHECK(
      ctx,
      resize_to_broadcast_target_size(a, b, cond, out) == Error::Ok,
      InvalidArgument,
      out);

  constexpr auto name = "where.self_out";

  ET_CHECK_MSG(
      cond_type == ScalarType::Bool || cond_type == ScalarType::Byte,
      "Unhandled dtype %s for where.self_out",
      torch::executor::toString(cond_type));
  

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
