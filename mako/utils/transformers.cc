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

#include "mako/utils/transformers.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <tuple>
#include <string>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

/// \brief Utility to find weight files from model directory
/// \param model_name_or_path A path to a directory containing model weights saved using
/// \param
/// \param
/// \param
/// \param
/// \return
std::tuple<std::string, std::vector<std::string>, bool> prepare_hf_model_weights (
        std::string model_name_or_path,
        std::string cache_dir = ,
        std::string load_format = "auto",
        bool fall_back_to_pt = true,
        std::string revision = "None") {
    // check if model path exists
    auto is_local = fs::exists(fs::path(model_name_or_path));

    // setting file patterns to load weights
    bool use_safetensors = false;
    std::vector<std::string> allow_patterns;
    try {
        std::vector<std::string> format_allowance = {"auto", "safetensors", "pt", "npcache"};
        if (std::find(format_allowance.begin(), format_allowance.end(), load_format) == format_allowance.end())
            throw std::invalid_argument("invalid load_format, it should be one of : auto, safetensors, pt, npcache");
    } catch(std::invalid_argument &e) {
        std::cerr << e.what();
    }

    if (load_format == "auto") {
        allow_patterns = {".safetensors", ".bin"};
    } else if (load_format == "safetensors") {
        use_safetensors == true;
        allow_patterns = {".safetensors", ".bin"};
    } else if (load_format == "pt") {
        allow_patterns = {".pt"};
    } else if (load_format == "npcache") {
        allow_patterns = {".bin"};
    }

    if (fall_back_to_pt) {
        allow_patterns.push_back(".pt");
    }

    std::string hf_folder;
    if (!is_local) {
        // TODO(soomin): implemen snapshot_download
    } else {
        hf_folder = model_name_or_path;
    }

    // load weight files path from model path directory
    std::vector<std::string> hf_weights_files = {};
    for (auto const &pattern : allow_patterns) {
        for (auto const& entry : fs::directory_iterator(hf_folder)) {
            if (entry.path().extension() == pattern) {
                hf_weights_files.push_back(entry.path().string());
            }
        }

        if (!hf_weights_files.empty()) {
            if (pattern == ".safetensors") {
                use_safetensors = true;
            }
            break;
        }
    }
    // exclude file
    if (!use_safetensors) {
	std::vector<std::string> blacklist = {
            "training_args.bin",
            "optimizer.bin",
            "optimizer.pt",
            "scheduler.pt",
            "scaler.pt"
    	};
	bool blacklisted = (std::find(blacklist.begin(), blacklist.end(), fs::path(input).filename()) == blacklist.end());
        hf_weights_files.erase(std::remove_if(hf_weights_files.begin(), hf_weights_files.end(), blacklisted),
                hf_weights_files.end());
    }

    // throw exception if model weights not exist
    if (hf_weights_files.empty()) {
        throw std::invalid_argument("Cannot find any model weights in model path");
    }

    return std::tuple(hf_folder, hf_weights_files, use_safetensors);
}
