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

Tensor& any_all_out(RuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, {}) == Error::Ok, InvalidArgument, out);

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();
  constexpr auto name = "any.all_out";


  return out;
}

Tensor& any_dims_out(
    RuntimeContext& ctx,
    const Tensor& in,
    optional<ArrayRef<int64_t>> dim_list,
    bool keepdim,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_reduction_args(in, dim_list, keepdim, {}, out),
      InvalidArgument,
      out);

  if (dim_list.has_value() && dim_list.value().empty()) {
    ET_KERNEL_CHECK(
        ctx, resize_tensor(out, in.sizes()) == Error::Ok, InvalidArgument, out);
  } else {
    ET_KERNEL_CHECK(
        ctx,
        resize_reduction_out(in, dim_list, keepdim, out) == Error::Ok,
        InvalidArgument,
        out);
  }

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();
  constexpr auto name = "any.dims_out";



  return out;
}

Tensor& any_out(
    RuntimeContext& ctx,
    const Tensor& in,
    int64_t dim,
    bool keepdim,
    Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx,
      check_reduction_args_single_dim(
          in, dim, keepdim, {}, out, /*allow_empty_dim*/ true),
      InvalidArgument,
      out);

  ET_KERNEL_CHECK(
      ctx,
      resize_reduction_out(in, dim, keepdim, out) == Error::Ok,
      InvalidArgument,
      out);

  ScalarType in_type = in.scalar_type();
  ScalarType out_type = out.scalar_type();
  constexpr auto name = "any.out";


  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
