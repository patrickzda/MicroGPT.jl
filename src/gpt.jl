# gpt.jl — a small GPT built on the vector/matrix`AValue` autograd engine
# (autograd.jl), the `Adam` optimizer (optimizer.jl), the `load_data` corpus
# loader (dataloader.jl) and the character `Tokenizer` (tokenizer.jl).

"""
    GPTConfig(; vocab_size, n_embd, n_head, n_layer, block_size)

Hyperparameters describing a GPT: vocabulary size, embedding width, number of
attention heads, number of transformer layers and the maximum sequence length.
"""
struct GPTConfig
    vocab_size::Int
    n_embd::Int
    n_head::Int
    n_layer::Int
    block_size::Int
end

GPTConfig(; vocab_size, n_embd, n_head, n_layer, block_size) =
    GPTConfig(vocab_size, n_embd, n_head, n_layer, block_size)

"""
    head_dim(config::GPTConfig)

Per-head embedding width, `n_embd ÷ n_head`.
"""
head_dim(config::GPTConfig) = config.n_embd ÷ config.n_head

"""
    GPT

A GPT model on the `AValue` engine: each parameter matrix is a single `AValue`
node. Holds its `GPTConfig`, a `state_dict` mapping names to weight matrices, a
flat `params` list for the optimizer, and the bound character `Tokenizer`.
"""
struct GPT
    config::GPTConfig
    state_dict::Dict{String,AValue}
    params::Vector{AValue}
    tokenizer::Tokenizer
end

# Random (nout, nin) matrix AValue, initialised ~ N(0, std).
param_matrix(nout, nin, std=0.08) = AValue(randn(nout, nin) .* std)

"""
    GPT(config::GPTConfig, tokenizer::Tokenizer; std=0.08)

Build a fresh GPT with randomly initialised weights bound to `tokenizer`. Each
parameter matrix is a single `AValue` node.
"""
function GPT(config::GPTConfig, tokenizer::Tokenizer; std=0.08)
    (; vocab_size, n_embd, n_layer, block_size) = config

    state_dict = Dict{String,AValue}(
        "wte" => param_matrix(vocab_size, n_embd, std),      # token embeddings
        "wpe" => param_matrix(block_size, n_embd, std),      # position embeddings
        "lm_head" => param_matrix(vocab_size, n_embd, std),  # output projection
    )
    for i in 1:n_layer
        state_dict["layer$i.attn_wq"] = param_matrix(n_embd, n_embd, std)
        state_dict["layer$i.attn_wk"] = param_matrix(n_embd, n_embd, std)
        state_dict["layer$i.attn_wv"] = param_matrix(n_embd, n_embd, std)
        state_dict["layer$i.attn_wo"] = param_matrix(n_embd, n_embd, std)
        state_dict["layer$i.mlp_fc1"] = param_matrix(4 * n_embd, n_embd, std)
        state_dict["layer$i.mlp_fc2"] = param_matrix(n_embd, 4 * n_embd, std)
    end

    return GPT(config, state_dict, collect(values(state_dict)), tokenizer)
end

# Fresh per-layer, per-head KV cache: keys[layer][head] is the list of cached
# head-sized key vectors (one `AValue` per past position).
kv_cache(cfg::GPTConfig) = [[AValue[] for _ in 1:cfg.n_head] for _ in 1:cfg.n_layer]

# Forward pass
function (model::GPT)(token_id, pos_id, keys, values)
    sd = model.state_dict
    cfg = model.config
    hd = head_dim(cfg)

    x = sd["wte"][token_id, :] + sd["wpe"][pos_id, :]  # joint token+position embedding
    x = rmsnorm(x)

    for li in 1:cfg.n_layer
        # 1) Multi-head attention block
        x_residual = x
        x = rmsnorm(x)
        q = sd["layer$li.attn_wq"] * x
        k = sd["layer$li.attn_wk"] * x
        v = sd["layer$li.attn_wv"] * x
        head_outs = AValue[]
        for h in 1:cfg.n_head
            hs = (h - 1) * hd
            push!(keys[li][h], k[hs+1:hs+hd])    # cache the keys
            push!(values[li][h], v[hs+1:hs+hd])  # cache the values
            K = hcat(keys[li][h]...)    # hd × t: cached keys as columns
            V = hcat(values[li][h]...)  # hd × t: cached values as columns
            attn_weights = softmax(transpose(K) * q[hs+1:hs+hd] / hd^0.5)
            push!(head_outs, V * attn_weights)  
        end
        x = sd["layer$li.attn_wo"] * vcat(head_outs...) + x_residual

        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = sd["layer$li.mlp_fc2"] * relu(sd["layer$li.mlp_fc1"] * x) + x_residual
    end

    return sd["lm_head"] * x   # vocab logits as one AValue
end

"""
    train!(model::GPT, docs; num_steps=1000, learning_rate=0.01,
           beta1=0.85, beta2=0.99, eps_adam=1e-8, verbose=true)

Train `model` in place on `docs` (a list of strings) with one `Adam` optimizer
over all weight matrices. Each step forwards one document, accumulating the
per-position cross-entropy loss, then backpropagates with `backward!` (recursive
topological walk — no tape) and takes one Adam step.
"""
function train!(model::GPT, docs;
    num_steps=1000, learning_rate=0.01,
    beta1=0.85, beta2=0.99, eps_adam=1e-8,
    verbose=true)
    cfg = model.config
    tok = model.tokenizer
    opt = Adam(model.params; α=learning_rate, β1=beta1, β2=beta2, ϵ=eps_adam)

    for step in 0:(num_steps-1)
        doc = docs[step%length(docs)+1]
        tokens = encode(tok, String(doc))
        n = min(cfg.block_size, length(tokens) - 1)

        # Forward the sequence, accumulating per-position cross-entropy losses
        keys, values = kv_cache(cfg), kv_cache(cfg)
        losses = AValue[]
        for pos_id in 1:n
            token_id, target_id = tokens[pos_id], tokens[pos_id+1]
            logits = model(token_id, pos_id, keys, values)
            probs = softmax(logits)
            push!(losses, -log(probs[target_id]))
        end
        loss = reduce(+, losses) / n   # average loss over the document

        backward!(loss)

        # Adam update with linear learning-rate decay
        opt.α = learning_rate * (1 - step / num_steps)
        step!(opt)
        zero_grad!(opt) # Reset the grads before the next step

        verbose && println("step $(lpad(step + 1, 4)) / $num_steps | loss $(round(loss.data[], digits=4))")
    end

    return model
end

"""
    generate(model::GPT; temperature=0.5) -> String

Sample one sequence from `model` and decode it to a String, stopping at the BOS
token or `block_size`. Probabilities come from one softmax `AValue` node.
"""
function generate(model::GPT; temperature=0.5)
    cfg = model.config
    tok = model.tokenizer
    BOS = tok.bos
    keys, values = kv_cache(cfg), kv_cache(cfg)

    token_id = BOS            # BOS is already a 1-based token id
    ids = Int[]               # sampled ids (1-based, fed straight to decode)
    for pos_id in 1:cfg.block_size
        logits = model(token_id, pos_id, keys, values)
        weights = softmax(logits / temperature).data

        # Sample an index from the categorical distribution.
        r = rand() * sum(weights)
        cw = 0.0
        token_id = cfg.vocab_size
        for idx in 1:cfg.vocab_size
            cw += weights[idx]
            if r <= cw
                token_id = idx
                break
            end
        end

        token_id == BOS && break
        push!(ids, token_id)
    end

    return decode(tok, ids)
end

# Persistence 
"""
    save_model(path, model::GPT, uchars::Vector{Char})

Save `model`: its config, raw weight matrices and the tokenizer vocabulary.
`load_model` rebuilds an equivalent `GPT` from the file.
"""
function save_model(path::AbstractString, model::GPT, uchars::Vector{Char})
    weights = Dict{String,Matrix{Float64}}(
        name => copy(t.data) for (name, t) in model.state_dict
    )
    serialize(path, (config=model.config, weights=weights, uchars=uchars))
    return path
end

"""
    load_model(path) -> GPT

Load a `GPT` saved by [`save_model`](@ref), reconstructing the tokenizer from
its stored vocabulary and wrapping each weight matrix back into an `AValue`.
"""
function load_model(path::AbstractString)
    obj = deserialize(path)
    uchars = obj.uchars
    char2id = Dict(c => i for (i, c) in enumerate(uchars))
    tokenizer = Tokenizer(uchars, char2id, length(uchars) + 1, length(uchars) + 1)

    state_dict = Dict{String,AValue}(name => AValue(w) for (name, w) in obj.weights)
    return GPT(obj.config, state_dict, collect(values(state_dict)), tokenizer)
end
