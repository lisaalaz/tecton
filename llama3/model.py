# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import re
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 8192
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        output = self.output(h).float()
        return output, h

class FunctionLM(nn.Module):
    def __init__(self, base_model, tokenizer, func_dict, load_path=None, inference_mode="func_embedding"):
        super().__init__()
        self.inference_mode = inference_mode
        self.model = base_model
        self.tokenizer = tokenizer
        self.func_dict = func_dict
        self.func_list = {v: k for k, v in func_dict.items()}
        # self.func_embed = ColumnParallelLinear(
        #     base_model.params.dim, len(func_list), bias=False, init_method=lambda x: x
        # )
        #self.errors = [] 

        # Make a linear layer with input dimension 4096 and output dimension the number of tools.
        # e.g. dim. 4096 x 13 for funcqa
        # This will be concatenated with (lm_head): Linear(in_features=4096, out_features=128256, bias=False):
        self.func_embed = nn.Linear(base_model.params.dim, len(func_dict), bias=False).to("cuda")

        # If the embeddings are already trained we load them:
        if load_path is not None and load_path != "None": # load func_embed weights
            embedding = torch.load(load_path)
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.to("cuda")
                embedding = {"weight": embedding}
            # truncate the embedding if necessary
            if embedding["weight"].shape[0] > len(func_dict):
                print(f"Truncated the function embedding from {embedding['weight'].shape[0]} to {len(func_dict)}")
                embedding["weight"] = embedding["weight"][:len(func_dict)]
            self.func_embed.load_state_dict(embedding)
        
        # set the basemodel to eval mode and freeze the weights
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.logits_bias = 0

    def set_bias(self, logits_bias):
        # This is applied to the toolkens only
        # Logit_bias is an optional parameter that modifies the likelihood of specified tokens appearing in a Completion. 
        # This parameter accepts a JSON object that maps tokens to an associated bias value from -100 (a ban) to 100 (exclusive selection of the token).
        self.logits_bias = logits_bias

    
    def get_vanilla_loss(self, raw_inputs):
        assert len(raw_inputs) == 1
        raw_inputs = raw_inputs[0]

        with torch.no_grad():
            raw_input_ids = torch.tensor(self.tokenizer.encode(raw_inputs, bos=True, eos=True))[:]
            labels = torch.tensor(self.tokenizer.encode(raw_inputs, bos=True, eos=True))[:]

            # Now inputs are the original token ids of the text input, without the eos token
            inputs = raw_input_ids[:-1].expand(1, -1).to("cuda")
            # and labels are the token ids with the above substitutions, without the bos token
            labels = labels[1:].expand(1, -1).to("cuda")

            # pass inputs (original token ids of the text input) through the base model and get the hidden states
            last_logits, h = self.model(inputs, 0) # h: (bsz, seqlen, dim)
            # and get the logits for every token position over the whole vocabulary
            token_logits = self.model.output(h) # (bsz, seqlen, vocab_size)
            # print(h.device)
        
        # finally compute the CE loss on this concatenated logits against the labels
        # (ie the tokenized inputs with function instead of result):
        loss = F.cross_entropy(token_logits.view(-1, token_logits.shape[-1]), labels.view(-1), ignore_index=-100)

        return loss



    def get_loss(self, raw_inputs, only_functoken=False):
        assert len(raw_inputs) == 1
        raw_inputs = raw_inputs[0]

        # inputs: starts with <bos>, ends without <eos>, (bsz, seqlen)
        # labels: starts without <bos>, ends with <eos>, (bsz, seqlen)
        with torch.no_grad():
            # prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=True) for x in raw_inputs]
            
            # Tokenize the input text. We keep a copy as input to the model (raw_input_ids),
            # and a copy as target to compute the loss (labels). We will mask the result with the (correct) tool name in the latter:
            raw_input_ids = torch.tensor(self.tokenizer.encode(raw_inputs["text"], bos=True, eos=True))[:]
            labels = torch.tensor(self.tokenizer.encode(raw_inputs["text"], bos=True, eos=True))[:]
            if "tar_eq" not in raw_inputs:
                raw_inputs["tar_eq"] = ["<" + raw_inputs["api"] + ">"]

            # Now take the first token where the result appears in the input (s),
            # the last token (t), and the target equation that is given in the input:
            for s, t, eq in zip(raw_inputs["start_token_idx_llama3"], raw_inputs["end_token_idx_llama3"], raw_inputs["tar_eq"]):
                
                # for different data formats
                # take just the operation from the target equation field, for example <log>:
                if "[" in eq:
                    op = re.search(r"(\[.*?\])", eq).group(1)
                elif "<" in eq:
                    op = re.search(r"(<.*?>)", eq).group(1)
                
                # if not found in the dictionary we eliminate the <> around it:
                if op not in self.func_dict:
                    op = op[1:-1]
                # Now mask the result:
                # replace the first result-containing token of the input with the id # of the func token
                # This is 128256 (size of word token vocabulary) + number of that tool in the tool dictionary

                labels[s] = self.func_dict[op] + 128256
                # and all the other result-containing tokens with -100 so they are ignored in training
                labels[s+1: t] = -100

            # labels = labels[1:]
            # Now inputs are the original token ids of the text input, without the eos token
            # When only_functoken is set to true all the other tokens that are not the tool token are also
            # masked out of the labels (and will be ignored in loss computation)
            if only_functoken: # This is False in the main paper results
                labels[labels < 128256] = -100 

            # Now inputs are the original token ids of the text input, without the eos token
            inputs = raw_input_ids[:-1].expand(1, -1).to("cuda")
            # and labels are the token ids with the above substitutions, without the bos token
            labels = labels[1:].expand(1, -1).to("cuda")
            
            # pass inputs (original token ids of the text input) through the base model and get the hidden states
            last_logits, h = self.model.forward(inputs, 0) # h: (bsz, seqlen, dim)
            # and get the logits for every token position over the whole vocabulary
            token_logits = self.model.output(h) # (bsz, seqlen, vocab_size)
            # print(h.device)
        
        # now take the hidden states output by the model and pass through the linear layer
        # to compute the logits over every token position over all the toolkens:
        func_logits = self.func_embed(h.float()) # (bsz, seqlen, len(func_list))
        
        # and concatenate the toolken logits after the vocabulary logits:
        concat_logits = torch.cat([token_logits, func_logits], dim=-1) # (bsz, seqlen, vocab_size + len(func_list))
        
        # finally compute the CE loss on this concatenated logits against the labels
        # (ie the tokenized inputs with function instead of result):
        loss = F.cross_entropy(concat_logits.view(-1, concat_logits.shape[-1]), labels.view(-1), ignore_index=-100)
        # check p, r, f1 for each function

        # show the predicted tokens
        pred = torch.argmax(concat_logits, dim=-1) # (bsz, seqlen)
        pred = pred.view(-1)
        # and target tokens (with substitution)
        labels = labels.view(-1)

        # for every tool, check if any target token is that tool (gives arrays of True/False):
        label_funcs = [labels == self.func_dict[op] + 128256 for op in self.func_dict.keys()]
        # for every tool, check if any predicted token is that tool (gives arrays of True/False):
        pred_funcs = [pred == self.func_dict[op] + 128256 for op in self.func_dict.keys()]
        
        # stack arrays
        label_funcs = torch.stack(label_funcs, dim=0)
        pred_funcs = torch.stack(pred_funcs, dim=0)
        # give arrays of tool set length with 0-1 values showing if tools are present (1) or not (0):
        tp = torch.sum(label_funcs * pred_funcs, dim=-1).detach().cpu().numpy()
        pred_funcs = torch.sum(pred_funcs, dim=-1).detach().cpu().numpy()
        true = torch.sum(label_funcs, dim=-1).detach().cpu().numpy()
        results = {
            "tp": tp,
            "pred": pred_funcs,
            "true": true
        }

        return loss, results
    


    @torch.no_grad()
    # generate is used at inference
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_token: List[int] = [], # 29897: ), 3892: )= 
        return_top: int = 0,
        disable_func: List[str] = [],
        disable_token: List[int] = [], # 29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        enable_only_token: List[int] = [],
        no_left_parens: bool = False,
        objs: List[str] = [],
    ) -> List[str]:
        #generate_log = []
        
        bsz = len(prompts)
        # print("objs", objs)

        obj_encodings = [self.tokenizer.encode("<"+obj+">", bos=False, eos=False)[1:-1] for obj in objs]
        # print("obj encoding", obj_encodings)

        assert bsz == 1 # we are only passing one prompt at a time at inference

        stop_token_substr = [torch.tensor(x).cuda().long() for x in stop_token if isinstance(x, list)]
        stop_token_single = [x for x in stop_token if isinstance(x, int)]
        
        func_list = list(self.func_dict.keys()) # list of available tools for current task

        # tokenize all the tools in func_list
        func_tokens = [self.tokenizer.encode(x[1:-1], bos=False, eos=False) for x in func_list]
        #generate_log.append(f"the func_tokens are: {func_tokens}\n")
        
        generation_log = [] # (token, [(token, logits, prob)])
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        # tokenize the input prompt
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts] # output: token ids list e.g. [306, 626, 9796]
        # min and max will be the same as we only have one tokenized prompt at inference
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size) # total generation length is the minimum of these two

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long() # make a tensor of shape bsz(=1) * total_len and fill it all with pad tokens
        
        for k, t in enumerate(prompt_tokens): # for index, item in prompt_tokens list
            tokens[k, : len(t)] = torch.tensor(t).long() # now replace the fist pad tokens in the newly created tokens tensor with the token ids of the prompt. Basically pad to max length.
        input_text_mask = tokens != self.tokenizer.pad_id # make a new equal size tensor and assign true to the positions in tokens that are not pad_tokens, and false everywhere else.
        start_pos = min_prompt_size # the start position equals the length of the shortest prompt (or the only prompt, in this case)
        prev_pos = 0
        hs = []
        
        for cur_pos in range(start_pos, total_len):

            # pass the tokens tensor from prev_pos (initialised 0) to cur_pos (initialised to be start_pos --> i.e. length of the prompt)
            # so basically initially you're just passing the prompt_tokens
            # the second arg of forward is start_pos (in this case 0)
            _, h = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            # get logits over the vocabulary for the word tokens
            logits_token = self.model.output(h[:, -1, :]).float() # (bsz, vocab_size)
            # get logits over the toolken matrix for the toolkens
            logits_func = self.func_embed(h[:, -1, :].float()) # (bsz, len(func_list))

            if self.inference_mode != "func_embedding":
                # if the task is not meant to produce toolkens (that is, when in tool mode to add the args!) we make a tensor of same size as logits func and fill it with -100,000 values
                # this effectively disables all the toolkens
                logits_func = torch.zeros_like(logits_func) - 1e5
            
            if len(enable_only_token) > 0:
                # if enable_only_token is not the empty list, we take the token positions in the word token logits tensor, 
                # and set all the others to -100,000, so that they are disabled and only the specified ones remain enabled
                disable_token = [x for x in list(range(logits_token.size()[1])) if x not in enable_only_token]
                logits_token[:, disable_token] = -1e5
            
            if len(disable_token) > 0:
                # if disable_token is not the empty list, we take the disabled token positions in the word token logits tensor, 
                # and set only those to -100,000, so that they are disabled
                logits_token[:, disable_token] = -1e5

            # topk: (bsz, 3)
            # print("after-topk", topk[1][0], [self.tokenizer.decode([x]) for x in topk[1][0].tolist()])

            for i, func in enumerate(disable_func): # for index, function in disable_func
                # get the token id of each function that must be disabled
                func_id = self.func_dict[func]
                # and set those functions to -100,000 in the logits tensor for the toolkens, so they are disabled
                logits_func[:, func_id] = -1e5

            # temperature can be set in inference_llama main with flag (default is 0)
            logits_func += self.logits_bias # positive logit bias (here usually in range 2.7-10) applied to func logits gives slightly more likelihood to generate tokens (max possible logit bias is 100)

            logits = torch.cat([logits_token, logits_func], dim=-1) # concatenate word and func logits
            
            # temperature can be set in inference_llama main with flag (default is 0)
            if temperature > 0:
                # sample with temperature
                probs = torch.softmax(logits / temperature, dim=-1)
                # the top-p param (default 0.95) maintains a balance between diversity and high-probability words by selecting tokens from the most probable tokens 
                # whose collective probability mass is greater than or equal to a threshold top-p.
                next_token = sample_top_p(probs, top_p)
            else:
                # if sample with temperature is disabled (ie. temperature=0) just take the argmax token
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(-1)


            # only replace token if the prompt is ended
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            if return_top > 0:
                # appends to the generation_log a tuple where the first element is the next token (previously chosen),
                # and the second element is a list of tuples where the first element is the index (ie. token idx) at which the next highest logit appears, and the second element is that logit).
                # This list has length equal to return_top (so we only take the return_top most likely tokens).
                if temperature > 0:
                    generation_log.append(
                    (next_token[0].item(), [(i.item(), logits[0, i.item()].item()) for i in torch.argsort(probs[0, :], descending=True)[:return_top]])
                  )
                else:
                    generation_log.append(
                    (next_token[0].item(), [(i.item(), logits[0, i.item()].item()) for i in torch.argsort(logits[0, :], descending=True)[:return_top]])
                  )
          
                
            
            # replace the next pad token in the tokens tensor after the tokens already replaced, with the next_token
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos # prev_pos moves forward to be the same as cur pos

            # if the next_token is a toolken or the next token is a stop token we passed when calling generate()
            # we break out of the generation loop
            if next_token[0] >= 128256 or next_token[0] in stop_token_single:
                # print("breaking!!")
                break
            
            # also if any of the generated tokens are equal to a stop token substring, break out of the generation loop
            if any([torch.equal(tokens[0, cur_pos - len(substr) + 1: cur_pos + 1], substr) for substr in stop_token_substr]):
                break


        decoded = []
        for i, t in enumerate(tokens.tolist()): # having ended generation, we enumerate index, tokens in the tokens tensor (should only be one tokens list)
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            
            # cur_pos is the position of the last generated token
            if t[cur_pos] >= 128256:
                if no_left_parens:
                    decoded.append(self.tokenizer.decode(t[:cur_pos]) + self.func_list[t[cur_pos] - 128256]) # decode with the toolken at cur_pos and append to decoded list
                else:
                    if "<" in self.func_list[0]:
                        decoded.append(self.tokenizer.decode(t[:cur_pos]) + self.func_list[t[cur_pos] - 128256] + "(") # same as above but also add "(" and append to decoded list
                    elif "[" in self.func_list[0]:
                        decoded.append(self.tokenizer.decode(t[:cur_pos]) + self.func_list[t[cur_pos] - 128256] + " <") # same but add " <" and append to decoded list
                    else:
                        raise NotImplementedError
            else:
                decoded.append(self.tokenizer.decode(t[:cur_pos + 1])) # if no toolken just append generation to decoded list
        
        # text_file = open("generate_log.txt", "a+")
        # for line in generate_log:
        #   text_file.write(line)
        # text_file.close()       
        if return_top > 0: # there is already a provision to return more than one token at each position
            return decoded, generation_log
        else:
            return decoded
        
    ####### start added code #######
    @torch.no_grad()
    def score(
        self,
        prompts: List[str],
        enable_only_token: List[int] = [],
    ) -> Dict[str, float]:
        #score_log = []
        
        bsz = len(prompts)
        # print("objs", objs)

        assert bsz == 1 # we are only passing one prompt at a time at inference
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        # tokenize the input prompt
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts] # output: token ids list e.g. [306, 626, 9796]
        # score_log.append(f"the prompt_tokens are {prompt_tokens}\n\n")
 
        # min and max will be the same as we only have one tokenized prompt at inference
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = 1 + max_prompt_size # we are scoring a single token

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long() # make a tensor of shape bsz(=1) * total_len and fill it all with pad tokens   
        
        for k, t in enumerate(prompt_tokens): # for index, item in prompt_tokens list
            tokens[k, : len(t)] = torch.tensor(t).long() # now replace the fist pad tokens in the newly created tokens tensor with the token ids of the prompt. Basically pad to max length.
        
        prev_pos = 0
        hs = []

        # pass the tokens tensor from prev_pos (initialised 0) to cur_pos (initialised to be start_pos --> i.e. length of the prompt)
        # so basically initially you're just passing the prompt_tokens
        # the second arg of forward is start_pos (in this case 0)
        _, h = self.model.forward(tokens, prev_pos)
        # score_log.append(f"h is {h}\n\n")

        # get logits over the vocabulary for the word tokens
        logits = self.model.output(h[:, -1, :]).float() # (bsz, vocab_size)
        # score_log.append(f"the logits are {logits}\n\n")
            
        if len(enable_only_token) > 0:
            # if enable_only_token is not the empty list, we take the token positions in the word token logits tensor, 
            # and set all the others to -100,000, so that they are disabled and only the specified ones remain enabled
            disable_token = [x for x in list(range(logits.size()[1])) if x not in enable_only_token]
            logits[:, disable_token] = -1e5
            
        # topk: (bsz, 3)
        # print("after-topk", topk[1][0], [self.tokenizer.decode([x]) for x in topk[1][0].tolist()])
            
        # temperature can be set in inference_llama main with flag (default is 0)
        probs = torch.softmax(logits, dim=-1)

        # score_log.append(f"the probs are {probs}\n\n")

        probs_list = probs.tolist()[0]

        #score_log.append(f"the probs_list is {probs_list}\n\n")

        probs_tokens = [probs_list[i] for i in enable_only_token]
        decoded_tokens = [self.tokenizer.decode([i]) for i in enable_only_token]
        out = dict(zip(decoded_tokens, probs_tokens))
        # score_log.append(f"the output to return is {out}\n\n")

        # text_file = open("score_log_funcmodel.txt", "a+")
        # for line in score_log:
        #   text_file.write(line)
        # text_file.close()       
       
        return out
    '''
    @torch.no_grad()
    # generate is used at inference
    def score(
        self,
        prompts: List[str],
        enable_only_token: List[int] = [],
    ) -> Dict[str, float]:
        score_log = []
        
        bsz = len(prompts)
        # print("objs", objs)

        assert bsz == 1 # we are only passing one prompt at a time at inference
        
        generation_log = [] # (token, [(token, logits, prob)])
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        # tokenize the input prompt
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts] # output: token ids list e.g. [306, 626, 9796]
        score_log.append(f"the prompt_tokens are {prompt_tokens}\n")
        text_file = open("score_log.txt", "a+")
        for line in score_log:
          text_file.write(line)
        text_file.close()    

        # min and max will be the same as we only have one tokenized prompt at inference
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = 1 + max_prompt_size # we are scoring a single token

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long() # make a tensor of shape bsz(=1) * total_len and fill it all with pad tokens
        score_log.append(f"the tokens are {tokens}\n")
        text_file = open("score_log.txt", "a+")
        for line in score_log:
          text_file.write(line)
        text_file.close()       
        
        for k, t in enumerate(prompt_tokens): # for index, item in prompt_tokens list
            score_log.append(f"k is {k} and t is {t}\n")
            text_file = open("score_log.txt", "a+")
            for line in score_log:
              text_file.write(line)
            text_file.close()    

            score_log.append(f"tokens[k, : len(t)] is {tokens[k, : len(t)]}\n")
            text_file = open("score_log.txt", "a+")
            for line in score_log:
              text_file.write(line)
            text_file.close()    

            tokens[k, : len(t)] = torch.tensor(t).long() # now replace the fist pad tokens in the newly created tokens tensor with the token ids of the prompt. Basically pad to max length.
        
        input_text_mask = tokens != self.tokenizer.pad_id # make a new equal size tensor and assign true to the positions in tokens that are not pad_tokens, and false everywhere else.
        start_pos = min_prompt_size # the start position equals the length of the shortest prompt (or the only prompt, in this case)
        prev_pos = 0
        hs = []

        score_log.append(f"the tokens NOW are {tokens}\n")
        text_file = open("score_log.txt", "a+")
        for line in score_log:
          text_file.write(line)
        text_file.close()       
        
        for cur_pos in range(start_pos, total_len):
            score_log.append("one iteration done!")
            text_file = open("score_log.txt", "a+")
            for line in score_log:
               text_file.write(line)
            text_file.close()       

            # pass the tokens tensor from prev_pos (initialised 0) to cur_pos (initialised to be start_pos --> i.e. length of the prompt)
            # so basically initially you're just passing the prompt_tokens
            # the second arg of forward is start_pos (in this case 0)
            _, h = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            score_log.append(f"h is {h}\n")
            text_file = open("score_log.txt", "a+")
            for line in score_log:
              text_file.write(line)
            text_file.close()       

            # get logits over the vocabulary for the word tokens
            logits_token = self.model.output(h[:, -1, :]).float() # (bsz, vocab_size)
            # get logits over the toolken matrix for the toolkens
            logits_func = self.func_embed(h[:, -1, :].float()) # (bsz, len(func_list))

            logits_func = torch.zeros_like(logits_func) - 1e5 # no function tokens
            
            if len(enable_only_token) > 0:
                # if enable_only_token is not the empty list, we take the token positions in the word token logits tensor, 
                # and set all the others to -100,000, so that they are disabled and only the specified ones remain enabled
                disable_token = [x for x in list(range(logits_token.size()[1])) if x not in enable_only_token]
                logits_token[:, disable_token] = -1e5
            
            # topk: (bsz, 3)
            # print("after-topk", topk[1][0], [self.tokenizer.decode([x]) for x in topk[1][0].tolist()])

            logits = torch.cat([logits_token, logits_func], dim=-1) # concatenate word and func logits
            
            # temperature can be set in inference_llama main with flag (default is 0)
            probs = torch.softmax(logits, dim=-1)
            
            
        probs_tokens = [probs[:, i] for i in enable_only_token]
        decoded_tokens = [self.tokenizer.decode(i) for i in enable_only_token]
        
        score_log.append(f"the probs_tokens are {probs_tokens}")
        score_log.append(f"the decoded_tokens are {decoded_tokens}")
        text_file = open("score_log.txt", "a+")
        for line in score_log:
          text_file.write(line)
        text_file.close()       

        out = dict(zip(decoded_tokens, probs_tokens))
        score_log.append(f"the output to return is {out}")

        text_file = open("score_log.txt", "a+")
        for line in score_log:
          text_file.write(line)
        text_file.close()       
       
        return out
    '''
        

    def func_decode(self, token_log, no_left_parens: bool = False):
        if no_left_parens:
                    decoded = self.tokenizer.decode([i[0] for i in token_log][:-1]) + self.func_list[[token_log][-1][-1][0] - 128256] # decode with the toolken at cur_pos and append to decoded list
        else:
            if "<" in self.func_list[0]:
                decoded = self.tokenizer.decode([i[0] for i in token_log][:-1]) + self.func_list[[token_log][-1][-1][0] - 128256] + "(" # add "(" and append to decoded string
            elif "[" in self.func_list[0]:
                decoded = self.tokenizer.decode([i[0] for i in token_log][:-1]) + self.func_list[[token_log][-1][-1][0] - 128256] + " <" # add " <" and append to decoded string
        return decoded
    
    def decode_list(self, token_list):
        if "<" in self.func_list[0]:
                decoded = self.tokenizer.decode(token_list[:-1]) + self.func_list[token_list[-1] - 128256] + "(" # add "(" and append to decoded string
        elif "[" in self.func_list[0]:
                decoded = self.tokenizer.decode(token_list[:-1]) + self.func_list[token_list[-1] - 128256] + " <" # add " <" and append to decoded string

        return decoded
    ######## end added code ########

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # sort probabilities in descending order
    probs_sum = torch.cumsum(probs_sort, dim=-1) # compute tensor of cumulative sums for each position in the sorted probs tensor
    mask = probs_sum - probs_sort > p # generate a mask tensor of Trues and Falses where every position where the item in probs_sum minus the item in probs_sort is greater than p we put True, else False
    probs_sort[mask] = 0.0 # assign zeros to all trues in the probs_sort tensor
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # basically replaces the only non-zero element with 1 dividing it by itself
    next_token = torch.multinomial(probs_sort, num_samples=1) # returns the index of the only 1 in the sorted tensor
    next_token = torch.gather(probs_idx, -1, next_token) # returns the index of the chosen token in the original tensor
    return next_token



class BaseLM(nn.Module):
    def __init__(self, base_model, tokenizer, load_path=None):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        # self.func_embed = ColumnParallelLinear(
        #     base_model.params.dim, len(func_list), bias=False, init_method=lambda x: x
        # )
        #self.errors = [] 

        # If the embeddings are already trained we load them:
        if load_path is not None and load_path != "None": # load func_embed weights
            embedding = torch.load(load_path)
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.to("cuda")
                embedding = {"weight": embedding}
        
        # set the basemodel to eval mode
        self.model.eval()

    
    def get_loss(self, raw_inputs):
        assert len(raw_inputs) == 1
        raw_inputs = raw_inputs[0]

        with torch.no_grad():
            raw_input_ids = torch.tensor(self.tokenizer.encode(raw_inputs, bos=True, eos=True))[:]
            labels = torch.tensor(self.tokenizer.encode(raw_inputs, bos=True, eos=True))[:]

            # Now inputs are the original token ids of the text input, without the eos token
            inputs = raw_input_ids[:-1].expand(1, -1).to("cuda")
            # and labels are the token ids with the above substitutions, without the bos token
            labels = labels[1:].expand(1, -1).to("cuda")

            # pass inputs (original token ids of the text input) through the base model and get the hidden states
            last_logits, h = self.model(inputs, 0) # h: (bsz, seqlen, dim)
            # and get the logits for every token position over the whole vocabulary
            token_logits = self.model.output(h) # (bsz, seqlen, vocab_size)
            # print(h.device)
        
        # finally compute the CE loss on this concatenated logits against the labels
        # (ie the tokenized inputs with function instead of result):
        loss = F.cross_entropy(token_logits.view(-1, token_logits.shape[-1]), labels.view(-1), ignore_index=-100)

        return loss
    
    @torch.no_grad()
    def score(
        self,
        prompts: List[str],
        enable_only_token: List[int] = [],
    ) -> Dict[str, float]:
        #score_log = []
        
        bsz = len(prompts)
        # print("objs", objs)

        assert bsz == 1 # we are only passing one prompt at a time at inference
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        # tokenize the input prompt
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts] # output: token ids list e.g. [306, 626, 9796]
        #score_log.append(f"the prompt_tokens are {prompt_tokens}\n\n")

        # min and max will be the same as we only have one tokenized prompt at inference
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = 1 + max_prompt_size # we are scoring a single token

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long() # make a tensor of shape bsz(=1) * total_len and fill it all with pad tokens   
        
        for k, t in enumerate(prompt_tokens): # for index, item in prompt_tokens list
            tokens[k, : len(t)] = torch.tensor(t).long() # now replace the fist pad tokens in the newly created tokens tensor with the token ids of the prompt. Basically pad to max length.
 
        prev_pos = 0
        hs = []

        # pass the tokens tensor from prev_pos (initialised 0) to cur_pos (initialised to be start_pos --> i.e. length of the prompt)
        # so basically initially you're just passing the prompt_tokens
        # the second arg of forward is start_pos (in this case 0)
        _, h = self.model.forward(tokens, prev_pos)
        #score_log.append(f"h is {h}\n\n")
        # get logits over the vocabulary for the word tokens
        logits = self.model.output(h[:, -1, :]).float() # (bsz, vocab_size)
        #score_log.append(f"the logits are {logits}\n\n")
            
        if len(enable_only_token) > 0:
            # if enable_only_token is not the empty list, we take the token positions in the word token logits tensor, 
            # and set all the others to -100,000, so that they are disabled and only the specified ones remain enabled
            disable_token = [x for x in list(range(logits.size()[1])) if x not in enable_only_token]
            logits[:, disable_token] = -1e5
            
        # topk: (bsz, 3)
        # print("after-topk", topk[1][0], [self.tokenizer.decode([x]) for x in topk[1][0].tolist()])
            
        # temperature can be set in inference_llama main with flag (default is 0)
        probs = torch.softmax(logits, dim=-1)

        # score_log.append(f"the probs are {probs}\n\n")
        # text_file = open("score_log_basemodel.txt", "a+")
        # for line in score_log:
        #     text_file.write(line)
        # text_file.close() 
            
        probs_list = probs.tolist()[0]

        # score_log.append(f"the probs_list is {probs_list}\n\n")
        # text_file = open("score_log_basemodel.txt", "a+")
        # for line in score_log:
        #     text_file.write(line)
        # text_file.close() 

        probs_tokens = [probs_list[i] for i in enable_only_token]
        decoded_tokens = [self.tokenizer.decode([i]) for i in enable_only_token]
        out = dict(zip(decoded_tokens, probs_tokens))
        #score_log.append(f"the output to return is {out}\n\n")

        # text_file = open("score_log_basemodel.txt", "a+")
        # for line in score_log:
        #   text_file.write(line)
        # text_file.close()       
       
        return out


    @torch.no_grad()
    # generate is used at inference
    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_token: List[int] = [], # 29897: ), 3892: )= 
        return_top: int = 0,
        disable_func: List[str] = [],
        disable_token: List[int] = [], # 29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        enable_only_token: List[int] = [],
        no_left_parens: bool = False,
        objs: List[str] = [],
    ) -> List[str]:
        #generate_log = []
        
        bsz = len(prompts)
        # print("objs", objs)

        assert bsz == 1 # we are only passing one prompt at a time at inference

        stop_token_substr = [torch.tensor(x).cuda().long() for x in stop_token if isinstance(x, list)]
        stop_token_single = [x for x in stop_token if isinstance(x, int)]
        
        generation_log = [] # (token, [(token, logits, prob)])
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)
        
        # tokenize the input prompt
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts] # output: token ids list e.g. [306, 626, 9796]
        # min and max will be the same as we only have one tokenized prompt at inference
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size) # total generation length is the minimum of these two

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long() # make a tensor of shape bsz(=1) * total_len and fill it all with pad tokens
        
        for k, t in enumerate(prompt_tokens): # for index, item in prompt_tokens list
            tokens[k, : len(t)] = torch.tensor(t).long() # now replace the fist pad tokens in the newly created tokens tensor with the token ids of the prompt. Basically pad to max length.
        input_text_mask = tokens != self.tokenizer.pad_id # make a new equal size tensor and assign true to the positions in tokens that are not pad_tokens, and false everywhere else.
        start_pos = min_prompt_size # the start position equals the length of the shortest prompt (or the only prompt, in this case)
        prev_pos = 0
        hs = []
        
        for cur_pos in range(start_pos, total_len):

            # pass the tokens tensor from prev_pos (initialised 0) to cur_pos (initialised to be start_pos --> i.e. length of the prompt)
            # so basically initially you're just passing the prompt_tokens
            # the second arg of forward is start_pos (in this case 0)
            _, h = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            # get logits over the vocabulary for the word tokens
            logits = self.model.output(h[:, -1, :]).float() # (bsz, vocab_size)
            
            if len(enable_only_token) > 0:
                # if enable_only_token is not the empty list, we take the token positions in the word token logits tensor, 
                # and set all the others to -100,000, so that they are disabled and only the specified ones remain enabled
                disable_token = [x for x in list(range(logits.size()[1])) if x not in enable_only_token]
                logits[:, disable_token] = -1e5
            
            if len(disable_token) > 0:
                # if disable_token is not the empty list, we take the disabled token positions in the word token logits tensor, 
                # and set only those to -100,000, so that they are disabled
                logits[:, disable_token] = -1e5

            # topk: (bsz, 3)
            # print("after-topk", topk[1][0], [self.tokenizer.decode([x]) for x in topk[1][0].tolist()])
            
            # temperature can be set in inference_llama main with flag (default is 0)
            if temperature > 0:
                # sample with temperature
                probs = torch.softmax(logits / temperature, dim=-1)
                # the top-p param (default 0.95) maintains a balance between diversity and high-probability words by selecting tokens from the most probable tokens 
                # whose collective probability mass is greater than or equal to a threshold top-p.
                next_token = sample_top_p(probs, top_p)
            else:
                # if sample with temperature is disabled (ie. temperature=0) just take the argmax token
                next_token = torch.argmax(logits, dim=-1)

            next_token = next_token.reshape(-1)


            # only replace token if the prompt is ended
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )

            if return_top > 0:
                # appends to the generation_log a tuple where the first element is the next token (previously chosen),
                # and the second element is a list of tuples where the first element is the index (ie. token idx) at which the next highest logit appears, and the second element is that logit).
                # This list has length equal to return_top (so we only take the return_top most likely tokens).
                if temperature > 0:
                    generation_log.append(
                    (next_token[0].item(), [(i.item(), logits[0, i.item()].item()) for i in torch.argsort(probs[0, :], descending=True)[:return_top]])
                  )
                else:
                    generation_log.append(
                    (next_token[0].item(), [(i.item(), logits[0, i.item()].item()) for i in torch.argsort(logits[0, :], descending=True)[:return_top]])
                  )
          
                
            
            # replace the next pad token in the tokens tensor after the tokens already replaced, with the next_token
            tokens[:, cur_pos] = next_token
            prev_pos = cur_pos # prev_pos moves forward to be the same as cur pos

            # if the next_token is a stop token we passed when calling generate()
            # we break out of the generation loop
            if next_token[0] in stop_token_single:
                # print("breaking!!")
                break
            
            # also if any of the generated tokens are equal to a stop token substring, break out of the generation loop
            if any([torch.equal(tokens[0, cur_pos - len(substr) + 1: cur_pos + 1], substr) for substr in stop_token_substr]):
                break


        decoded = []
        for i, t in enumerate(tokens.tolist()): # having ended generation, we enumerate index, tokens in the tokens tensor (should only be one tokens list)
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.tokenizer.eos_id)]
            except ValueError:
                pass
            
            # cur_pos is the position of the last generated token
            decoded.append(self.tokenizer.decode(t[:cur_pos + 1])) # if no toolken just append generation to decoded list
        
        # text_file = open("generate_log.txt", "a+")
        # for line in generate_log:
        #   text_file.write(line)
        # text_file.close()       
        if return_top > 0: # there is already a provision to return more than one token at each position
            return decoded, generation_log
        else:
            return decoded
        

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True) # sort probabilities in descending order
    probs_sum = torch.cumsum(probs_sort, dim=-1) # compute tensor of cumulative sums for each position in the sorted probs tensor
    mask = probs_sum - probs_sort > p # generate a mask tensor of Trues and Falses where every position where the item in probs_sum minus the item in probs_sort is greater than p we put True, else False
    probs_sort[mask] = 0.0 # assign zeros to all trues in the probs_sort tensor
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # basically replaces the only non-zero element with 1 dividing it by itself
    next_token = torch.multinomial(probs_sort, num_samples=1) # returns the index of the only 1 in the sorted tensor
    next_token = torch.gather(probs_idx, -1, next_token) # returns the index of the chosen token in the original tensor
    return next_token