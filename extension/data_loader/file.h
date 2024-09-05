/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/runtime/platform/compiler.h>

#ifndef _WIN32

#include <sys/mman.h>
#include <unistd.h>

ET_INLINE long get_os_page_size() {
  return sysconf(_SC_PAGESIZE)
}

#else

#define NOMINMAX
#include <windows.h>

#include <executorch/runtime/platform/unistd_pal.h>

// TODO: This is MIT licensed - check for compatibility with BSD.
#include <executorch/extension/data_loader/mman_windows.h>

ET_INLINE long get_os_page_size() {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  long pagesize = si.dwAllocationGranularity > si.dwPageSize ? si.dwAllocationGranularity : si.dwPageSize;
  return pagesize;
}

#endif
