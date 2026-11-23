# tiny-voca-lg from the command line

Luganda text to a WAV file with Node or Bun, from a GGUF file on
[Hugging Face](https://huggingface.co/henryhale/tiny-voca-lg/tree/main/gguf). No packages.

```bash
node js/say.mjs "Oli otya? Nkwagala nnyo." --voice F2 --output out.wav --type q8_0
```

`--type` is `q4_0` (3.3 MB), `q8_0` (5.5 MB), `f16` (9.2 MB) or `f32` (18.2 MB);
`--file` takes a local path or a URL instead.

It uses the same code as the [web demo](../web/): the model in
[`web/public/tiny-voca.js`](../web/public/tiny-voca.js) (from upstream
[tiny-voca](https://github.com/n4ze3m/tiny-voca/tree/main/js), unchanged) and the
front end in [`web/public/lug.js`](../web/public/lug.js). `text.js` cuts the text
into sentences and writes the WAV.

To make the GGUF files: `python scripts/export_gguf.py --checkpoint weights/tiny-voca-lug.pt`.
