// Copyright 2024 The Mako Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <optional>
#include <string>

#include <absl/container/flat_hash_map.h>
#include <absl/strings/string_view.h>
#include <torch/torch.h>

#include "mako/utils/export.h"

namespace mako {
namespace utils {
namespace torch {
/// \brief Alternative to ``torch::load``, equivalent to PyTorch's ``torch.load``.
///  Loads an object saved with ``torch.save`` from a file.
/// \param f A string view containing a file name.
/// \param map_location How to remap storage locations.
/// \param weights_only Indicates whether unpickler should be restricted to loading only tensors.
/// \param mmap Indicates whether the file should be mmaped rather than loading all the storages into memory.
/// \return Pairs of name and weight of the loaded model.
absl::flat_hash_map<std::string, ::torch::Tensor> MAKO_API load(
  absl::string_view f,
  std::optional<::torch::Device> map_location = std::nullopt,
  bool weights_only                           = true,
  bool mmap                                   = false);
} // namespace torch
} // namespace utils
} // namespace mako
