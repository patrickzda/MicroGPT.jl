# Usage (from the project root):
#     julia --project=. run.jl
#
# On first run the names dataset is downloaded to `input.txt` automatically.

using MicroGPT

docs = load_data("input.txt")
tokenizer = Tokenizer(docs)
println("num docs: $(length(docs)) | vocab size: $(tokenizer.vocab_size)")

config = GPTConfig(
    vocab_size = tokenizer.vocab_size,
    n_embd     = 64,
    n_head     = 4,
    n_layer    = 4,
    block_size = 32,
)
model = GPT(config, tokenizer)

train!(model, docs; num_steps = 2000, learning_rate = 0.01)

println("\nSamples:")
for _ in 1:20
    println("  ", generate(model; temperature = 0.8))
end

save_model("model.jls", model, tokenizer.uchars)
println("\nSaved trained model to model.jls")
new_model = load_model("model.jls")


