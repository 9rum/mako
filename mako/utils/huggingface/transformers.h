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
#include <utility>

#include <absl/strings/string_view.h>
#include <boost/coroutine2/all.hpp>
#include <torch/torch.h>

#include "mako/utils/export.h"

namespace mako {
namespace utils {
inline namespace huggingface {
/// \brief Utility to download and initialize Hugging Face Transformers model.
/// \param model_name_or_path A path to a directory containing model weights saved using ``save_pretrained``.
/// \param cache_dir Path to the folder where cached files are stored.
/// \param load_format Format of the model to load.
///  Must be one of ``"auto"``, ``"safetensors"``, ``"pt"``, or ``"npcache"``.
/// \param fall_back_to_pt If ``true``, will always allow pt format.
/// \param revision An optional Git revision id which can be a branch name, a tag, or a commit hash.
/// \return An iterator generating the pairs of name and weight of the loaded model.
boost::coroutines2::coroutine<std::pair<absl::string_view, torch::Tensor>>::pull_type MAKO_API weight_iterator(
  absl::string_view model_name_or_path,
  std::optional<absl::string_view> cache_dir = std::nullopt,
  absl::string_view load_format              = "auto",
  bool fall_back_to_pt                       = true,
  std::optional<absl::string_view> revision  = std::nullopt);
} // namespace huggingface
} // namespace utils
} // namespace mako
