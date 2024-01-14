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

// From C10's macro definition:
//
// Definition of an adaptive XX_API macro, that depends on whether you are
// building the library itself or not, routes to XX_EXPORT and XX_IMPORT.
// Basically, you will need to do this for each shared library that you are
// building, and the instruction is as follows: assuming that you are building
// a library called libawesome.so. You should:
// (1) for your cmake target (usually done by "add_library(awesome, ...)"),
//     define a macro called AWESOME_BUILD_MAIN_LIB using
//     target_compile_options.
// (2) define the AWESOME_API macro similar to the one below.
// And in the source file of your awesome library, use AWESOME_API to
// annotate public symbols.
//
// Since Mako is designed to be a serving system, it always builds an
// executable file; in other words, MAKO_BUILD_MAIN_LIB will always be defined
// (if any), thus here we will not define complicated macros but only a
// compiler-adaptive macro for public symbol annotation purpose.
#ifdef __GNUC__
#define MAKO_API __attribute__((__visibility__("default")))
#else
#ifdef _MSC_VER
#define MAKO_API __declspec(dllexport)
#else
#define MAKO_API
#endif // _MSC_VER
#endif // __GNUC__
