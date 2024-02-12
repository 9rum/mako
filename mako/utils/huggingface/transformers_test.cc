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

#include "mako/utils/huggingface/transformers.h"

#include <vector>
#include <map>
#include <array>

#include <torch/torch.h>
#include <gtest/gtest.h>

TEST(TransformersTest, test) {
    //Load vLLM's weight iterator output
    std::map<std::string, std::vector<int>> vllm_output;
    vllm_output["model.layers.11.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.11.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.11.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.11.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.12.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.12.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.12.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.12.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.12.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.12.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.12.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.12.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.12.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.12.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.13.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.13.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.13.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.13.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.13.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.13.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.13.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.13.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.13.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.13.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.14.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.14.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.14.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.14.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.14.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.14.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.14.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.14.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.14.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.14.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.15.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.15.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.15.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.15.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.15.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.15.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.15.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.15.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.15.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.15.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.16.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.16.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.16.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.16.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.16.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.16.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.16.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.16.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.16.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.16.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.17.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.17.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.17.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.17.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.17.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.17.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.17.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.17.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.17.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.17.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.18.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.18.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.18.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.18.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.18.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.18.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.18.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.18.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.18.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.18.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.19.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.19.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.19.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.19.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.19.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.19.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.19.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.19.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.19.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.19.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.20.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.20.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.20.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.20.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.20.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.20.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.20.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.20.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.20.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.20.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.21.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.21.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.21.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.21.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.21.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.21.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.21.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.21.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.21.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.21.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.22.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.22.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.22.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.22.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.22.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.22.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.22.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.22.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.22.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.22.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.23.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.23.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.23.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.23.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.23.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.23.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.23.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.23.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.23.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.23.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.24.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.24.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.24.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.24.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.24.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.24.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.24.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.24.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.24.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.24.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.25.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.25.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.25.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.25.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.25.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.25.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.25.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.25.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.25.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.25.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.26.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.26.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.26.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.26.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.26.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.26.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.26.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.26.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.26.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.26.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.27.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.27.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.27.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.27.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.27.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.27.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.27.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.27.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.27.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.27.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.28.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.28.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.28.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.28.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.28.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.28.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.28.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.28.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.28.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.28.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.29.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.29.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.29.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.29.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.29.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.29.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.29.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.29.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.29.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.29.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.30.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.30.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.30.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.30.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.30.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.30.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.30.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.30.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.30.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.30.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.31.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.31.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.31.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.31.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.31.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.31.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.31.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.31.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.31.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.31.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.norm.weight"] = {4096}
    vllm_output["lm_head.weight"] = {32000, 4096}
    vllm_output["model.embed_tokens.weight"] = {32000, 4096}
    vllm_output["model.layers.0.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.0.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.0.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.0.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.0.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.0.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.0.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.0.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.0.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.0.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.1.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.1.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.1.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.1.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.1.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.1.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.1.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.1.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.1.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.1.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.2.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.2.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.2.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.2.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.2.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.2.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.2.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.2.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.2.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.2.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.3.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.3.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.3.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.3.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.3.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.3.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.3.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.3.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.3.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.3.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.4.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.4.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.4.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.4.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.4.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.4.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.4.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.4.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.4.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.4.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.5.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.5.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.5.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.5.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.5.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.5.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.5.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.5.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.5.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.5.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.6.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.6.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.6.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.6.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.6.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.6.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.6.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.6.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.6.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.6.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.7.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.7.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.7.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.7.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.7.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.7.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.7.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.7.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.7.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.7.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.8.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.8.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.8.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.8.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.8.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.8.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.8.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.8.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.8.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.8.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.9.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.9.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.9.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.9.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.9.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.9.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.9.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.9.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.9.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.9.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.10.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.10.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.10.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.10.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.10.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.10.mlp.gate_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.10.mlp.up_proj.weight"] = {11008, 4096}
    vllm_output["model.layers.10.mlp.down_proj.weight"] = {4096, 11008}
    vllm_output["model.layers.10.input_layernorm.weight"] = {4096}
    vllm_output["model.layers.10.post_attention_layernorm.weight"] = {4096}
    vllm_output["model.layers.11.self_attn.q_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.11.self_attn.k_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.11.self_attn.v_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.11.self_attn.o_proj.weight"] = {4096, 4096}
    vllm_output["model.layers.11.self_attn.rotary_emb.inv_freq"] = {64}
    vllm_output["model.layers.11.mlp.gate_proj.weight"] = {11008, 4096}

    auto iterator = mako::utils::huggingface::weight_iterator("/DATA/base_models/Llama-2-7b-hf");
    for (i = vllm_output.start(); i = vllm_output.end(); i++) {
        std::pair(std::string, torch::Tensor) value = iterator();
        // check weight name is in vllm's weights' name
        it = vllm_output.find(value.fisrt);
        ASSERT_NE(it, vllm_output.end()) << "INVALID WEIGHT NAME LOADED";

        // check tensor's shape is equal
        ASSERT_EQ(vllm_output[value.first], value.second.sizes()) << "INVALID WEIGHT SHAPE LOADED";

        // check all weight's are loaded
        vllm_output.erase(value.first);
        ASSERT_TRUE(vllm_output.empty()) << "NOT ALL WEIGHTS LOADED";
    }
}

int main() {
    testing::InitGoogleTest();
    return RUN_ALL_TESTS();
}