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

#include "mako/utils/huggingface/hub.h"

#ifdef _WIN32
#include <cstdlib>
#else
#include <pwd.h>
#include <unistd.h>
#endif // _WIN32

#include <filesystem>

namespace fs = std::filesystem;

/// \brief Utility to find the user’s home directory, equivalent to Python's ``pathlib.Path.home``.
/// \return A string view to the home directory.
static inline absl::string_view home() {
  #ifdef _WIN32
  return std::getenv("USERPROFILE");
  #else
  return getpwuid(getuid())->pw_dir;
  #endif // _WIN32
}

/// \brief Alternative to ``std::getenv``, equivalent to Python's ``os.getenv``.
/// \param __key The name of environment variable.
/// \param __default Default value returned if ``__key`` does not exist.
/// \return Value of the environment variable ``__key``.
static inline absl::string_view _getenv(absl::string_view __key, absl::string_view __default) {
  auto value = std::getenv(__key.data());
  return value ? value : __default;
}

const auto _default_home                 = fs::path(home()) / fs::path(".cache");
absl::string_view default_home           = _default_home.string();
absl::string_view _xdg_cache_home        = _getenv("XDG_CACHE_HOME", default_home);
const auto __hf_home                     = fs::path(_xdg_cache_home) / fs::path("huggingface");
absl::string_view _hf_home               = _getenv("HF_HOME", __hf_home.string());
const auto _default_cache_path           = fs::path(_hf_home) / fs::path("hub");
absl::string_view default_cache_path     = _default_cache_path.string();
absl::string_view _huggingface_hub_cache = _getenv("HUGGINGFACE_HUB_CACHE", default_cache_path);
absl::string_view _hf_hub_cache          = _getenv("HF_HUB_CACHE", _huggingface_hub_cache);
absl::string_view _default_revision      = "main";
