# gpt2-haskell

A hasktorch implementation of GPT2 matching [Karpathy's mingpt
implementation](https://github.com/karpathy/minGPT/). Only inference
for now.

## Getting Started

1. Install Nix
2. Enable Flakes
3. Download the `model.safetensors` from [HuggingFace](https://huggingface.co/openai-community/gpt2)

### Run the inference example

1. `nix develop .`
2. `cabal run inference -- [absolute/path/to/model.safetensors]`

You should see an output similar to the following:

```
Input: Hello, I'm a language model,
Generated text:
Just "Hello, I'm a language model, which means I have a set of features for each"
```

You can edit `./examples/inference/Main.hs` to change the input or
number of tokens generated.

## Developing

1. `nix develop .`

## TODO

- Training
- Constraints on safetensors functions
- Test larger gpt2 models
- Test GPU
- Implement logging
