# Copyright 2024 The Mako Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# For PyTorch extensions, version should be greater than 3.13.
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(mako LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD    11 CACHE STRING "The C standard whose features are requested to build this target.")
set(CMAKE_CXX_STANDARD  17 CACHE STRING "The C++ standard whose features are requested to build this target.")
set(CMAKE_CUDA_STANDARD 17 CACHE STRING "The CUDA standard whose features are requested to build this target.")
set(CMAKE_C_STANDARD_REQUIRED    ON)
set(CMAKE_CXX_STANDARD_REQUIRED  ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# If PyTorch was installed via conda or pip, CMAKE_PREFIX_PATH
# can be queried using torch.utils.cmake_prefix_path variable.
# Find LibTorch from virtual environment to avoid tedious setup.
execute_process(COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
string(APPEND CMAKE_C_FLAGS    " ${TORCH_C_FLAGS}")
string(APPEND CMAKE_CXX_FLAGS  " ${TORCH_CXX_FLAGS}")
string(APPEND CMAKE_CUDA_FLAGS " ${TORCH_CUDA_FLAGS}")

add_subdirectory(mako)
