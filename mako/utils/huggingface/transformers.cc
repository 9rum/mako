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
#include <cassert>
#include <filesystem>
#include <fstream>

#include <absl/strings/str_format.h>
#include <boost/bind/bind.hpp>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;

using nlohmann::json;

static inline std::tuple<std::string, std::vector<std::string>, bool> prepare_load(
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
    allow_patterns  = {".safetensors"};
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

  std::string hf_folder;
  if (!is_local) {
    // Download model weights from Hugging Face Hub.
    // TODO: implement ``snapshot_download``
    throw std::logic_error("snapshot download is currently not supported");
  } else {
    hf_folder = model_name_or_path;
  }

  // If a weight file is stored as a string_view, it outlives the actual string value.
  // To avoid lifetime issue, use string as an exception for weight files.
  //
  // NOTE:
  //
  // Due to lifetime issues, a string_view is usually a poor choice for a return value
  // and almost always a poor choice for a data member.
  // https://abseil.io/docs/cpp/guides/strings
  std::vector<std::string> hf_weight_files;
  for (auto pattern : allow_patterns) {
    for (const auto &entry : fs::directory_iterator(hf_folder)) {
      if (entry.path().extension().compare(pattern) == 0) {
        hf_weight_files.push_back(entry.path().string());
      }
    }
    if (!hf_weight_files.empty()) {
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
    hf_weight_files.erase(std::remove_if(
      hf_weight_files.begin(),
      hf_weight_files.end(),
      [&](absl::string_view entry){
        for (auto suffix: blacklist) {
          if (suffix.size() <= entry.size() && entry.compare(entry.size()-suffix.size(), suffix.size(), suffix) == 0) {
            return true;
          }
        }
        return false;
      }), hf_weight_files.end());
  }

  if (hf_weight_files.empty()) {
    throw std::runtime_error(absl::StrFormat("Cannot find any model weights with %s", model_name_or_path));
  }

  return std::make_tuple(hf_folder, hf_weight_files, use_safetensors);
}

static inline void load(
  boost::coroutines2::coroutine<std::pair<std::string, torch::Tensor>>::push_type &yield,
  absl::string_view model_name_or_path,
  std::optional<absl::string_view> cache_dir,
  absl::string_view load_format,
  bool fall_back_to_pt,
  std::optional<absl::string_view> revision) {
  auto [hf_folder, hf_weight_files, use_safetensors] = prepare_load(
    model_name_or_path,
    cache_dir,
    load_format,
    fall_back_to_pt,
    revision);

  if (load_format.compare("npcache") == 0) {
    // Currently npcache only supports .bin checkpoints.
    assert(!use_safetensors);

    // Convert the model weights from torch tensors to numpy arrays for faster loading.
    auto np_folder = fs::path(hf_folder) / fs::path("np");
    fs::create_directories(np_folder);

    auto weight_names_file = np_folder / fs::path("weight_names.json");
    // Use file lock to prevent multiple processes from dumping the same model weights to numpy at the same time.
    // TODO: use file lock
    if (!fs::exists(weight_names_file)) {
      std::vector<std::string> weight_names;
      for (const auto &file : hf_weight_files) {
        auto stream = std::ifstream(file, std::ios::binary);
        auto buf    = std::vector<char>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
        stream.close();
        auto weights = torch::pickle_load(buf).toGenericDict();
        for (const auto &weight : weights) {
          auto name       = weight.key().toStringRef();
          auto param_path = np_folder / fs::path(name);
          // TODO: np.save(param_path, param.cpu().detach().numpy())
          weight_names.push_back(name);
        }
      }
      std::ofstream(weight_names_file) << json(weight_names);
    }

    auto weight_names = json::parse(std::ifstream(weight_names_file)).get<std::vector<std::string>>();
    for (const auto &name : weight_names) {
      auto param_path = np_folder / fs::path(name);
      // TODO: yield torch.from_numpy(np.load(param_path))
    }
  } else if (use_safetensors) {
    // TODO: implement ``safe_open``
    throw std::logic_error("safetensors is currently not supported");
  } else {
    for (const auto &file : hf_weight_files) {
      auto stream = std::ifstream(file, std::ios::binary);
      auto buf    = std::vector<char>(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
      stream.close();

      // CAUTION:
      //
      // Libtorch version 2.1.2 and few other previous versions may result in errors in torch::load_pickle().
      // Some versions of Libtorch lacks support for the recent header information associated with Pickle's protocols.
      //
      // We suggest you to use Libtorch 2.2.0, but if you want to use other versions(2.1.2 and etc.),
      // transform your pickle file(.bin) into hexadecimal form and
      // check the ZIP file headaer part ends with '80 02 7d 71 00 28 58 19 00 00 00'.
      //
      // Examples:
      // A file with the following binary header is acceptable in version 2.1.2:    But this file will cause an error:
      // 000000: 50 4b 03 04 00 00 08 08 00 00 00 00 00 00 00 00  PK..............  000000: 50 4b 03 04 00 00 08 08 00 00 00 00 00 00 00 00  PK..............
      // 000010: 00 00 00 00 00 00 00 00 00 00 25 00 3d 00 70 79  ..........%.=.py  000010: 00 00 00 00 00 00 00 00 00 00 25 00 3d 00 70 79  ..........%.=.py
      // 000020: 74 6f 72 63 68 5f 6d 6f 64 65 6c 2d 30 30 30 30  torch_model-0000  000020: 74 6f 72 63 68 5f 6d 6f 64 65 6c 2d 30 30 30 30  torch_model-0000
      // 000030: 31 2d 6f 66 2d 30 30 30 30 33 2f 64 61 74 61 2e  1-of-00003/data.  000030: 32 2d 6f 66 2d 30 30 30 30 33 2f 64 61 74 61 2e  2-of-00003/data.
      // 000040: 70 6b 6c 46 42 39 00 5a 5a 5a 5a 5a 5a 5a 5a 5a  pklFB9.ZZZZZZZZZ  000040: 70 6b 6c 46 42 39 00 5a 5a 5a 5a 5a 5a 5a 5a 5a  pklFB9.ZZZZZZZZZ
      // 000050: 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a  ZZZZZZZZZZZZZZZZ  000050: 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a  ZZZZZZZZZZZZZZZZ
      // 000060: 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a  ZZZZZZZZZZZZZZZZ  000060: 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a  ZZZZZZZZZZZZZZZZ
      // 000070: 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a  ZZZZZZZZZZZZZZZZ  000070: 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a 5a  ZZZZZZZZZZZZZZZZ
      // 000080: 80 02 7d 71 00 28 58 19 00 00 00                 ..}q.(X....       000080: 80 02 7d 71 00 28 58 22 00 00 00                 ..}q.(X"...
      auto weights = torch::pickle_load(buf).toGenericDict();
      for (const auto &weight : weights) {
        yield(std::make_pair(weight.key().toStringRef(), weight.value().toTensor()));
      }
    }
  }
}

boost::coroutines2::coroutine<std::pair<std::string, torch::Tensor>>::pull_type mako::utils::huggingface::weight_iterator(
  absl::string_view model_name_or_path,
  std::optional<absl::string_view> cache_dir,
  absl::string_view load_format,
  bool fall_back_to_pt,
  std::optional<absl::string_view> revision) {
  boost::coroutines2::coroutine<std::pair<std::string, torch::Tensor>>::pull_type iterator{
    boost::bind(
      load,
      boost::placeholders::_1,
      model_name_or_path,
      cache_dir,
      load_format,
      fall_back_to_pt,
      revision)};
    return iterator;
}
