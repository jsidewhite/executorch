/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdint>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/copy_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;
using TensorList = exec_aten::TensorList;

/**
 * unbind_copy.int_out(Tensor input, int dim=0, *, Tensor(a!)[] out) -> ()
 */
void unbind_copy_int_out(
    RuntimeContext& ctx,
    const Tensor& input,
    int64_t dim,
    TensorList out) {
  (void)ctx;
  // Support python-style negative indexing.
  if (dim < 0) {
    dim += input.dim();
  }

  ET_KERNEL_CHECK(
      ctx, check_unbind_copy_args(input, dim, out), InvalidArgument, );

  if (input.numel() == 0) {
    return;
  }

  const size_t leading_dims = getLeadingDims(input, dim);
  const size_t trailing_dims = getTrailingDims(input, dim);
  const size_t step = input.size(dim) * trailing_dims;

  ScalarType in_type = input.scalar_type();
  ScalarType out_type = out[0].scalar_type();

  
}

} // namespace native
} // namespace executor
} // namespace torch
