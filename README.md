
<div align="center">
<img src="https://i.postimg.cc/kg3gqwB1/Gemini-Generated-Image-bws52zbws52zbws5.png"alt="Go Llama" style="max-width: 600px; width: 100%;">
</div>
This repo is a rewrite by (mostly) hand of karpathys [nanoGPT] (https://github.com/karpathy/nanogpt) but in GO

This is purely just to help me understand inference and transformers better


Not optimal at all, just for learning. Meant to be used as a hackable super simple project to learn how LLM inference works from a first principles perspective

This is purely inference, no training (yet)


## Implemented
1) Tokenizer
    - The tokenizer is the translator sitting beteen human text and numberr the model can process
    ```
    "Hello, world!"  →  [Tokenizer.Encode]  →  [15496, 11, 995, 0]  →  fed into the GPT model
    ```