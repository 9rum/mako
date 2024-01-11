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

#include <filesystem>
#include <optional>
#include <pwd.h>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

#include <torch/torch.h>

#include "mako/utils/export.h"

namespace fs = std::filesystem;

/// \brief Alternative to ``std::getenv``, equivalent to Python's ``os.getenv``.
/// \param __key The name of environment variable.
/// \param __default Default value returned if ``__key`` does not exist.
/// \return Value of the environment variable ``__key``.
inline std::string _getenv(const char *__key, std::string __default) {
  const char *value = std::getenv(__key);
  return value ? std::string(value) : __default;
}

// Definitions of constants and macros to interact with Hugging Face Hub.
// Most of the below constants/macros are adapted from
// https://github.com/huggingface/huggingface_hub/blob/v0.20.0/src/huggingface_hub/constants.py.
auto _default_home          = fs::path(getpwuid(getuid())->pw_dir) / fs::path(".cache");
auto default_home           = _default_home.string();
auto _xdg_cache_home        = _getenv("XDG_CACHE_HOME", default_home);
auto __hf_home              = fs::path(_xdg_cache_home) / fs::path("huggingface");
auto _hf_home               = _getenv("HF_HOME", __hf_home.string());
auto _default_cache_path    = fs::path(_hf_home) / fs::path("hub");
auto default_cache_path     = _default_cache_path.string();
auto _huggingface_hub_cache = _getenv("HUGGINGFACE_HUB_CACHE", default_cache_path);
auto _hf_hub_cache          = _getenv("HF_HUB_CACHE", _huggingface_hub_cache);
auto _default_revision      = std::string("main");

#define HF_HOME               _hf_home
#define HUGGINGFACE_HUB_CACHE _huggingface_hub_cache
#define HF_HUB_CACHE          _hf_hub_cache
#define DEFAULT_REVISION      _default_revision

namespace mako {
namespace utils {
/// \brief Utility to download and initialize Hugging Face Transformers model.
/// \param model_name_or_path A path to a directory containing model weights saved using ``save_pretrained``.
/// \param cache_dir Path to the folder where cached files are stored.
/// \param load_format Format of the model to load. Must be one of ``"auto"``, ``"safetensors"``, ``"pt"``, or ``"npcache"``.
/// \param fall_back_to_pt If ``true``, will always allow pt format.
/// \param revision An optional Git revision id which can be a branch name, a tag, or a commit hash.
/// \return Pairs of name and weight of the loaded model.
std::vector<std::tuple<std::string, torch::Tensor>> MAKO_API hf_model_weights(
  std::string model_name_or_path,
  std::optional<std::string> cache_dir = std::nullopt,
  std::string load_format              = "auto",
  bool fall_back_to_pt                 = true,
  std::optional<std::string> revision  = std::nullopt);
} // namespace utils
} // namespace mako
