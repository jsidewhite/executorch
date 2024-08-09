/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <tuple>

#include <executorch/kernels/portable/cpu/util/normalization_ops_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using Tensor = exec_aten::Tensor;

std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_no_training_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const exec_aten::optional<Tensor>& weight,
    const exec_aten::optional<Tensor>& bias,
    const Tensor& running_mean,
    const Tensor& running_var,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& invstd_out) {
  (void)ctx;

  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(out, mean_out, invstd_out);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx, resize_tensor(mean_out, {0}) == Error::Ok, InvalidArgument, ret_val);

  ET_KERNEL_CHECK(
      ctx,
      resize_tensor(invstd_out, {0}) == Error::Ok,
      InvalidArgument,
      ret_val);

  ET_KERNEL_CHECK(
      ctx,
      check_batch_norm_args(
          in,
          weight,
          bias,
          running_mean,
          running_var,
          momentum,
          eps,
          out,
          mean_out,
          invstd_out),
      InvalidArgument,
      ret_val);

  // For now, only support the contiguous dim order
  ET_KERNEL_CHECK(
      ctx,
      is_contiguous_dim_order(in.dim_order().data(), in.dim_order().size()),
      InvalidArgument,
      ret_val);

  size_t C_dim = in.dim() >= 1 ? 1 : 0;
  size_t C = in.size(C_dim);
  size_t outer = getLeadingDims(in, C_dim);
  size_t inner = getTrailingDims(in, C_dim);

  constexpr auto name = "native_batch_norm_legit_no_training.out";

  
  return ret_val;
}

std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const exec_aten::optional<Tensor>& weight,
    const exec_aten::optional<Tensor>& bias,
    Tensor& running_mean,
    Tensor& running_var,
    bool training,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& invstd_out) {
  (void)ctx;

  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(out, mean_out, invstd_out);

  ET_KERNEL_CHECK_MSG(
      ctx,
      training == false,
      InvalidArgument,
      ret_val,
      "Portable kernels only support inference mode!");

  return _native_batch_norm_legit_no_training_out(
      ctx,
      in,
      weight,
      bias,
      running_mean,
      running_var,
      momentum,
      eps,
      out,
      mean_out,
      invstd_out);
}

std::tuple<Tensor&, Tensor&, Tensor&> _native_batch_norm_legit_no_stats_out(
    RuntimeContext& ctx,
    const Tensor& in,
    const exec_aten::optional<Tensor>& weight,
    const exec_aten::optional<Tensor>& bias,
    bool training,
    double momentum,
    double eps,
    Tensor& out,
    Tensor& mean_out,
    Tensor& invstd_out) {
  (void)ctx;
  (void)in;
  (void)weight;
  (void)bias;
  (void)momentum;
  (void)eps;

  std::tuple<Tensor&, Tensor&, Tensor&> ret_val(out, mean_out, invstd_out);

  ET_KERNEL_CHECK_MSG(
      ctx,
      training == false,
      InvalidArgument,
      ret_val,
      "Portable kernels only support inference mode!");

  ET_KERNEL_CHECK_MSG(
      ctx,
      training == true,
      InvalidArgument,
      ret_val,
      "running_mean & running_var must be provided during inference!");

  return ret_val;
}

} // namespace native
} // namespace executor
} // namespace torch
