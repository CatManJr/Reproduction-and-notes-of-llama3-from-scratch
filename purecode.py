from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch
import torch.nn.functional as F
import json

tokenizer_path = "Meta-Llama-3-8B/tokenizer.model" # 使用官方预训练的tokenizer use your local address of the tokenizer model

special_tokens = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|reserved_special_token_2|>",
            "<|reserved_special_token_3|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|reserved_special_token_4|>",
            "<|eot_id|>", # 结束符 end of turn 特别地定义前5个reserved_special_token，循环定义剩余的251个特殊符号
        ] + [f"<|reserved_special_token_{i}|>" for i in range(5, 256 - 5)]
mergeable_ranks = load_tiktoken_bpe(tokenizer_path) # Byte Pair Encoding (BPE) ranks from the tokenizer model
tokenizer = tiktoken.Encoding(
    name = Path(tokenizer_path).name, # 使用tokenizer文件名作为tokenizer的名字
    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1, 3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\s)|\s+", # 使用官方的正则表达式
    mergeable_ranks= mergeable_ranks, # 使用官方的BPE ranks
    special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}, # 使用官方的special_tokens
)

with open("Meta-Llama-3-8B/params.json", "r") as f:
    config = json.load(f)
# use your local address of the params

'''
Use the model params to load the model
the model has 32 transformer layers
each multi-head attention block has 32 heads
the vocab size and so on
'''
dim = config["dim"]
n_layers = config["n_layers"]
n_heads = config["n_heads"]
n_kv_heads = config["n_kv_heads"]
vocab_size = config["vocab_size"]
multiple_of = config["multiple_of"]
ffn_dim_multiplier = config["ffn_dim_multiplier"]
norm_eps = config["norm_eps"]
rope_theta = torch.tensor(config["rope_theta"])

model = torch.load("Meta-Llama-3-8B/consolidated.00.pth") # 使用官方预训练的模型参数 use your local address of the pth

# text 2 tokens
# prompt = "to be or not to be, that is the question"
prompt = "To be or not to be, that is the"
tokens = [128000] + tokenizer.encode(prompt)
print(tokens)
tokens = torch.tensor(tokens)
prompt_split_as_tokens = [tokenizer.decode([token.item()]) for token in tokens]
print(prompt_split_as_tokens)

#convert tokens to their embedding
embedding_layer = torch.nn.Embedding(vocab_size, dim)
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"]) # model
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)

# RMS Normalization
def rms_norm(tensor, norm_weights):
    return (tensor * 
            torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) 
                        + norm_eps)) * norm_weights
    
token_embeddings = rms_norm(token_embeddings_unnormalized, model["layers.0.attention_norm.weight"])
q_layer0 = model["layers.0.attention.wq.weight"]
head_dim = q_layer0.shape[0] // n_heads
q_layer0 = q_layer0.reshape(n_heads, head_dim, dim)

# first head of the first layer
q_layer0_head0 = q_layer0[0]


# multiply the query weights with the token embedding, to recive a query for the token
q_per_token = torch.matmul(token_embeddings, q_layer0_head0.T)

# positioning encoding
#Rotary Positional Embeddings
# use theta degree to position the token's poiton in the query
#eg: dog = 1 theta, The dog = 2 theta

q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
# split the query vectors into pairs to apply a rotational angle shift to each pair
# torch.Size([17, 64, 2])128 length queries split into 64 pairs for each token in the prompt
# each of those 64 pairs will be rotated by m*(theta)
# where m is the position of the token for which we are rotating the query

# using dot product of complex numbers to rotate a vector
zero_to_one_split_into_64_parts = torch.tensor(range(64))/64
freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)

freqs_for_each_token = torch.outer(torch.arange(token_embeddings_unnormalized.shape[0]), freqs)
# token_embeddings_unnormalized.shape[0] tokens in the prompt
freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

# convert the query to complex numbers and then dot product to rotate the query
q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)

q_per_token_as_complex_numbers_rotated = q_per_token_as_complex_numbers * freqs_cis

q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers_rotated)
q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

# Keys
'''
> keys generate key vectors also of dimention 128
> keys have only 1/4th the number of the weights as queries, this is because the weights for keys are shared across 4 heads at a time, to reduce the number of computations need
> keys are also rotated to add positional info, just like queries because of the same reasons
'''
k_layer0 = model["layers.0.attention.wk.weight"]
k_layer0 = k_layer0.view(n_kv_heads, k_layer0.shape[0] // n_kv_heads, dim)
k_layer0_head0 = k_layer0[0]
k_per_token = torch.matmul(token_embeddings, k_layer0_head0.T)
k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)

# multiply the queries and key matrices
qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(head_dim)**0.5


# make mask of future tokens
mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
mask = torch.triu(mask, diagonal=1)

qk_per_token_after_masking = qk_per_token + mask

qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)

'''
scores (0-1) are used to determine how much of value matrix is used per token
> just like keys, value weights are also shared acorss every 4 attention heads (to save computation)
> as a result, the shape of the value weight matrix below is [8x128x4096]
'''
v_layer0 = model["layers.0.attention.wv.weight"]
v_layer0 = v_layer0.view(n_kv_heads, v_layer0.shape[0] // n_kv_heads, dim)

v_layer0_head0 = v_layer0[0]

# vector of values
v_per_token = torch.matmul(token_embeddings, v_layer0_head0.T)

# attention(Z) = softmax(QK^T/sqrt(d_k))V
qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)

# run a loop and perform the exact same math as the cells above but for every head in the first layer
qkv_attention_store = []

for head in range(n_heads):
    q_layer0_head = q_layer0[head]
    k_layer0_head = k_layer0[head//4] # key weights are shared across 4 heads
    v_layer0_head = v_layer0[head//4] # value weights are shared across 4 heads
    q_per_token = torch.matmul(token_embeddings, q_layer0_head.T)
    k_per_token = torch.matmul(token_embeddings, k_layer0_head.T)
    v_per_token = torch.matmul(token_embeddings, v_layer0_head.T)

    q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
    q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
    q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

    k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
    k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
    k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis[:len(tokens)])
    k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)

    qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
    mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
    mask = torch.triu(mask, diagonal=1)
    qk_per_token_after_masking = qk_per_token + mask
    qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
    qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
    qkv_attention_store.append(qkv_attention)

# len(qkv_attention_store) = 32

# merge all attention scores into one large matrix
stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)

# Weights matrix
w_layer0 = model["layers.0.attention.wo.weight"]

# matmul the simple linear layer of weights
embedding_delta = torch.matmul(stacked_qkv_attention, w_layer0.T)

# adding the change in the embedding value after attention, to the original token embeddings
embedding_after_edit = token_embeddings_unnormalized + embedding_delta

# rms normalization
# run a feed forward neural network through the embedding delta
embedding_after_edit_normalized = rms_norm(embedding_after_edit, model["layers.0.ffn_norm.weight"])

# loading the ff weights
# implementing the feed forward network
w1 = model["layers.0.feed_forward.w1.weight"]
w2 = model["layers.0.feed_forward.w2.weight"]
w3 = model["layers.0.feed_forward.w3.weight"]
output_after_feedforward = torch.matmul(F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)

'''
in llama3, they used a SwiGLU feedforward network, 
this network architecture is really good at adding non linearity 
when needed by the model.
its pretty standard to use this feed forward network architecture 
in llms these days
'''

'''
embedding_after_edit_normalized 是输入的标准化后的嵌入向量，形状为 [12, 4096]。
w1, w2, w3 是从模型中加载的权重，假设每个权重的形状为 [input_dim, hidden_dim]。
F.silu 是 PyTorch 中的 Swish 激活函数。
torch.matmul 是 PyTorch 中的矩阵乘法函数，用于执行线性变换。
'''
# Loop the first layer
layer_0_embedding = embedding_after_edit+output_after_feedforward
# 32 layers
final_embedding = token_embeddings_unnormalized
for layer in range(n_layers):
    qkv_attention_store = []
    layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"])
    q_layer = model[f"layers.{layer}.attention.wq.weight"]
    q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
    k_layer = model[f"layers.{layer}.attention.wk.weight"]
    k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
    v_layer = model[f"layers.{layer}.attention.wv.weight"]
    v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    for head in range(n_heads):
        q_layer_head = q_layer[head]
        k_layer_head = k_layer[head//4]
        v_layer_head = v_layer[head//4]
        q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
        k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
        v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)
        q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
        q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
        q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)
        k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
        k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
        k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)
        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T)/(128)**0.5
        mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        qk_per_token_after_masking = qk_per_token + mask
        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1).to(torch.bfloat16)
        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
        qkv_attention_store.append(qkv_attention)

    stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)
    embedding_after_edit = final_embedding + embedding_delta
    embedding_after_edit_normalized = rms_norm(embedding_after_edit, model[f"layers.{layer}.ffn_norm.weight"])
    w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
    w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
    w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
    output_after_feedforward = torch.matmul(torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) * torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
    final_embedding = embedding_after_edit+output_after_feedforward

# rms norm
final_embedding = rms_norm(final_embedding, model["norm.weight"])

# predict and decode
logits = torch.matmul(final_embedding[-1], model["output.weight"].T)
next_token = torch.argmax(logits, dim=-1)

word = tokenizer.decode([next_token.item()])
print(f'Llama‘s prediction:\n {word}\n')
print(prompt+word)