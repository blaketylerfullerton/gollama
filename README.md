This repo is a rewrite by (mostly) hand of karpathys nanoGPT but in GO

This is purely just to help me understand inference and transformers better


Not optimal at all, just for learning

This is purely inference, no training (yet)


## Implemented
1) Tokenizer
    - The tokenizer is the translator sitting beteen human text and numberr the model can process
    ```
    "Hello, world!"  →  [Tokenizer.Encode]  →  [15496, 11, 995, 0]  →  fed into the GPT model
    ```