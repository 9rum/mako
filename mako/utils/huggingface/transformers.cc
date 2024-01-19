// Adapted from https://github.com/vllm-project/vllm/blob/v0.2.7/vllm/model_executor/weight_utils.py
// Copyright 2024 The Mako Authors
// Copyright 2023 The vLLM team
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

#include "mako/utils/huggingface/transformers.h"

#include <algorithm>
#include <filesystem>
#include <fstream>

#include <absl/strings/str_format.h>

namespace fs = std::filesystem;

static inline std::tuple<absl::string_view, std::vector<absl::string_view>, bool> prepare_load(
  absl::string_view model_name_or_path,
  std::optional<absl::string_view> cache_dir,
  absl::string_view load_format,
  bool fall_back_to_pt,
  std::optional<absl::string_view> revision) {
  auto is_local        = fs::is_directory(fs::path(model_name_or_path));
  auto use_safetensors = false;

  // Some quantized models use .pt files for storing the weights.
  std::vector<absl::string_view> allow_patterns;
  if (load_format.compare("auto") == 0) {
    allow_patterns = {".safetensors", ".bin"};
  } else if (load_format.compare("safetensors") == 0) {
    use_safetensors = true;
    allow_patterns  = {".safetensors", ".bin"};
  } else if (load_format.compare("pt") == 0) {
    allow_patterns = {".pt"};
  } else if (load_format.compare("npcache") == 0) {
    allow_patterns = {".bin"};
  } else {
    throw std::invalid_argument(absl::StrFormat("Unknown load format: %s", load_format));
  }

  if (fall_back_to_pt) {
    allow_patterns.push_back(".pt");
  }

  absl::string_view hf_folder;
  if (!is_local) {
    // Download model weights from Hugging Face Hub.
    // TODO(soomin): implement ``snapshot_download``
    throw std::logic_error("snapshot download is currently not supported");
  } else {
    hf_folder = model_name_or_path;
  }

  std::vector<absl::string_view> hf_weights_files;
  for (const auto &pattern : allow_patterns) {
    for (const auto &entry : fs::directory_iterator(hf_folder)) {
      if (entry.path().extension().compare(pattern) == 0) {
        hf_weights_files.push_back(entry.path().string());
      }
    }
    if (!hf_weights_files.empty()) {
      if (pattern.compare(".safetensors") == 0) {
        use_safetensors = true;
      }
      break;
    }
  }

  // Exclude files that are not needed for inference.
  // https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
  if (!use_safetensors) {
    std::vector<absl::string_view> blacklist = {
      "training_args.bin",
      "optimizer.bin",
      "optimizer.pt",
      "scheduler.pt",
      "scaler.pt",
    };
    hf_weights_files.erase(std::remove_if(
      hf_weights_files.begin(),
      hf_weights_files.end(),
      [&](absl::string_view entry){
        for (auto suffix: blacklist) {
          if (suffix.size() <= entry.size() && entry.compare(entry.size()-suffix.size(), suffix.size(), suffix) == 0) {
            return true;
          }
        }
        return false;
      }), hf_weights_files.end());
  }

  if (hf_weights_files.empty()) {
    throw std::runtime_error(absl::StrFormat("Cannot find any model weights with %s", model_name_or_path));
  }

  return std::make_tuple(hf_folder, hf_weights_files, use_safetensors);
}

static inline void load(
  boost::coroutines2::coroutine<std::tuple<absl::string_view, torch::Tensor>>::push_type &yield,
  absl::string_view model_name_or_path,
  std::optional<absl::string_view> cache_dir,
  absl::string_view load_format,
  bool fall_back_to_pt,
  std::optional<absl::string_view> revision) {
  auto [hf_folder, hf_weights_files, use_safetensors] = prepare_load(
    model_name_or_path,
    cache_dir,
    load_format,
    fall_back_to_pt,
    revision);

  if (load_format.compare("npcache") == 0) {
    throw std::logic_error("npcache is currently not supported");
  }

  if (use_safetensors) {
    throw std::logic_error("safetensors is currently not supported");
  }

  for (auto file : hf_weights_files) {
    auto stream = std::ifstream(file.data(), std::ios::binary);
    auto buf    = std::vector<char>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
    stream.close();
    auto weights = torch::pickle_load(buf).toGenericDict();
    for (const auto &weight : weights) {
      yield(std::make_tuple(weight.key().toStringRef(), weight.value().toTensor()));
    }
  }
}

boost::coroutines2::coroutine<std::tuple<absl::string_view, torch::Tensor>>::pull_type mako::utils::huggingface::weight_iterator(
  absl::string_view model_name_or_path,
  std::optional<absl::string_view> cache_dir,
  absl::string_view load_format,
  bool fall_back_to_pt,
  std::optional<absl::string_view> revision) {
  boost::coroutines2::coroutine<std::tuple<absl::string_view, torch::Tensor>>::pull_type iterator{
    std::bind(
      load,
      std::placeholders::_1,
      model_name_or_path,
      cache_dir,
      load_format,
      fall_back_to_pt,
      revision)};
    return iterator;
}
