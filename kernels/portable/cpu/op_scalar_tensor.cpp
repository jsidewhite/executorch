/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/kernels/portable/cpu/scalar_utils.h>
#include <executorch/runtime/kernel/kernel_includes.h>

namespace torch {
namespace executor {
namespace native {

Tensor& scalar_tensor_out(RuntimeContext& ctx, const Scalar& s, Tensor& out) {
  (void)ctx;

  ET_KERNEL_CHECK(
      ctx, resize_tensor(out, {}) == Error::Ok, InvalidArgument, out);

  ScalarType s_type = utils::get_scalar_dtype(s);
  ScalarType out_type = out.scalar_type();

  constexpr auto name = "scalar_tensor.out";

  
  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
