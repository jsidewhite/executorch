/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <executorch/extension/module/module.h>

#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/data_loader/mmap_data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <executorch/runtime/platform/runtime.h>

/**
 * Unwrap a Result to obtain its value (direct object, not a pointer).
 * If the Result contains an error, propagate the error via trivial function
 * return. The macro wraps the object into a unique_ptr.
 *
 * Note: A function using ET_UNWRAP_UNIQUE should itself return a Result or
 * Error.
 *
 * @param[in] result__ Expression yielding the result to unwrap.
 */
#define ET_TAKE_WRAPPED_UNIQUE(et_result__)                           \
  (std::make_unique<std::remove_reference_t<decltype(*et_result__)>>( \
        std::move(*et_result__)))

#define ET_UNWRAP_UNIQUE_OR_RETURN(outvar__, result__) \
  do {                                                 \
    auto et_result__ = (result__);                     \
    if (!et_result__.ok()) {                           \
      return et_result__.error();                      \
    }                                                  \
    outvar__ = ET_TAKE_WRAPPED_UNIQUE(et_result__);    \
  } while (false)

using ::exec_aten::Tensor;
using ::executorch::extension::FileDataLoader;
using ::executorch::extension::MallocMemoryAllocator;
using ::executorch::extension::MmapDataLoader;
using ::executorch::runtime::DataLoader;
using ::executorch::runtime::Error;
using ::executorch::runtime::EValue;
using ::executorch::runtime::EventTracer;
using ::executorch::runtime::HierarchicalAllocator;
using ::executorch::runtime::MemoryAllocator;
using ::executorch::runtime::MemoryManager;
using ::executorch::runtime::MethodMeta;
using ::executorch::runtime::Program;
using ::executorch::runtime::Result;
using ::executorch::runtime::Span;

namespace executorch {
namespace extension {

Module::Module(
    const std::string& file_path,
    const Module::LoadMode load_mode,
    std::unique_ptr<EventTracer> event_tracer)
    : file_path_(file_path),
      load_mode_(load_mode),
      memory_allocator_(std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)) {
  ::executorch::runtime::runtime_init();
}

Module::Module(
    std::unique_ptr<DataLoader> data_loader,
    std::unique_ptr<MemoryAllocator> memory_allocator,
    std::unique_ptr<MemoryAllocator> temp_allocator,
    std::unique_ptr<EventTracer> event_tracer)
    : data_loader_(std::move(data_loader)),
      memory_allocator_(
          memory_allocator ? std::move(memory_allocator)
                           : std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(
          temp_allocator ? std::move(temp_allocator)
                         : std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)) {
  ::executorch::runtime::runtime_init();
}

Module::Module(
    std::shared_ptr<Program> program,
    std::unique_ptr<MemoryAllocator> memory_allocator,
    std::unique_ptr<MemoryAllocator> temp_allocator,
    std::unique_ptr<EventTracer> event_tracer)
    : program_(std::move(program)),
      memory_allocator_(
          memory_allocator ? std::move(memory_allocator)
                           : std::make_unique<MallocMemoryAllocator>()),
      temp_allocator_(
          temp_allocator ? std::move(temp_allocator)
                         : std::make_unique<MallocMemoryAllocator>()),
      event_tracer_(std::move(event_tracer)) {
  ::executorch::runtime::runtime_init();
}

Error Module::load(const Program::Verification verification) {
  if (!is_loaded()) {
    if (!data_loader_) {
      switch (load_mode_) {
        case LoadMode::File:
          ET_UNWRAP_UNIQUE_OR_RETURN(data_loader_, FileDataLoader::from(file_path_.c_str()));
          break;
        case LoadMode::Mmap:
            ET_UNWRAP_UNIQUE_OR_RETURN(data_loader_, MmapDataLoader::from(
              file_path_.c_str(), MmapDataLoader::MlockConfig::NoMlock));
          break;
        case LoadMode::MmapUseMlock:
            ET_UNWRAP_UNIQUE_OR_RETURN(data_loader_, MmapDataLoader::from(file_path_.c_str()));
          break;
        case LoadMode::MmapUseMlockIgnoreErrors:
            ET_UNWRAP_UNIQUE_OR_RETURN(data_loader_, MmapDataLoader::from(
              file_path_.c_str(),
              MmapDataLoader::MlockConfig::UseMlockIgnoreErrors));
          break;
      }
    };
    auto wrapped_program = Program::load(data_loader_.get(), verification);
    ET_CHECK_UNWRAPPABLE(wrapped_program);
    auto program = ET_TAKE_WRAPPED_UNIQUE(wrapped_program);
    program_ = std::shared_ptr<Program>(
        program.release(),
        [data_loader = std::move(data_loader_)](Program* pointer) {
          delete pointer;
        });
  }
  return Error::Ok;
}

bool Module::is_loaded() const {
  return program_ != nullptr;
}

Result<std::unordered_set<std::string>> Module::method_names() {
  ET_CHECK_OK_OR_RETURN_ERROR(load());
  const auto method_count = program_->num_methods();
  std::unordered_set<std::string> result;
  result.reserve(method_count);

  for (auto index = 0; index < method_count; ++index) {
    result.emplace(program_->get_method_name(index).get());
  }
  return result;
}

Error Module::load_method(const std::string& method_name) {
  if (!is_method_loaded(method_name)) {
    ET_CHECK_OK_OR_RETURN_ERROR(load());

    MethodHolder method_holder;
    auto wrapped_method_metadata =
        program_->method_meta(method_name.c_str());
    ET_CHECK_UNWRAPPABLE(wrapped_method_metadata);
    const auto method_metadata = std::move(*wrapped_method_metadata);
    const auto planned_buffersCount =
        method_metadata.num_memory_planned_buffers();
    method_holder.planned_buffers.reserve(planned_buffersCount);
    method_holder.planned_spans.reserve(planned_buffersCount);

    for (auto index = 0; index < planned_buffersCount; ++index) {
      const auto buffer_size =
          method_metadata.memory_planned_buffer_size(index).get();
      method_holder.planned_buffers.emplace_back(buffer_size);
      method_holder.planned_spans.emplace_back(
          method_holder.planned_buffers.back().data(), buffer_size);
    }
    method_holder.planned_memory = std::make_unique<HierarchicalAllocator>(Span(
        method_holder.planned_spans.data(),
        method_holder.planned_spans.size()));
    method_holder.memory_manager = std::make_unique<MemoryManager>(
        memory_allocator_.get(),
        method_holder.planned_memory.get(),
        temp_allocator_.get());
    ET_UNWRAP_UNIQUE_OR_RETURN(method_holder.method, program_->load_method(
        method_name.c_str(),
        method_holder.memory_manager.get(),
        event_tracer_.get()));
    methods_.emplace(method_name, std::move(method_holder));
  }
  return Error::Ok;
}

bool Module::is_method_loaded(const std::string& method_name) const {
  return methods_.count(method_name);
}

Result<MethodMeta> Module::method_meta(const std::string& method_name) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  return methods_.at(method_name).method->method_meta();
}

Result<std::vector<EValue>> Module::execute(
    const std::string& method_name,
    const std::vector<EValue>& input) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method(method_name));
  auto& method = methods_.at(method_name).method;

  for (auto index = 0; index < input.size(); ++index) {
    ET_CHECK_OK_OR_RETURN_ERROR(method->set_input(input[index], index));
  }
  ET_CHECK_OK_OR_RETURN_ERROR(method->execute());

  const auto outputs_size = method->outputs_size();
  std::vector<EValue> outputs(outputs_size);
  ET_CHECK_OK_OR_RETURN_ERROR(
      method->get_outputs(outputs.data(), outputs_size));

  return outputs;
}

Error Module::set_output_data_ptr(Tensor& output_tensor, size_t output_index) {
  ET_CHECK_OK_OR_RETURN_ERROR(load_method("forward"));
  auto& method = methods_.at("forward").method;
  return method->set_output_data_ptr(
      output_tensor.mutable_data_ptr(), output_tensor.nbytes(), output_index);
}

} // namespace extension
} // namespace executorch
