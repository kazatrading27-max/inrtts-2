~/index-tts/checkpoints/
├── bpe.model                     (745KB)   ✅ unchanged
├── bpe_28000.model               (834KB)   ✅ unchanged
├── config.yaml                   (2.5KB)   ✅ unchanged
├── feat1.pt                      (57KB)    ✅ unchanged
├── feat2.pt                      (374KB)   ✅ unchanged
├── gpt.pth                       (3.6GB)   ✅ original preserved (untouched)
├── gpt/                                    ✅ NEW — created by conversion!
│   ├── model.safetensors         (~3.6GB)  ← YOUR finetuned GPT (HF format)
│   ├── config.json               (~1KB)    ← GPT2LMHeadModel config
│   ├── tokenizer.json            (1.29MB)  ← GPT2 tokenizer
│   ├── tokenizer_config.json     (26B)     ← tokenizer config
│   └── preprocessor_config.json  (342B)    ← audio preprocessor config
├── hf_cache/                               ✅ unchanged
├── pinyin.vocab                  (9KB)     ✅ unchanged
├── qwen0.6bemo4-merge/                     ✅ already HF format (unchanged)
│   ├── model.safetensors
│   ├── config.json
│   └── ...
├── s2mel.pth                     (1.2GB)   ✅ unchanged
└── wav2vec2bert_stats.pt         (9KB)     ✅ unchanged