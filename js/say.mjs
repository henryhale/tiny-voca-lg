// Luganda text to a WAV file with Node or Bun, from a GGUF file. No packages.
//
//   node js/say.mjs "Oli otya? Nkwagala nnyo." --voice F2 --output out.wav [--type q8_0] [--file path-or-url]

import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { readGGUF, TinyVoca } from "../web/public/tiny-voca.js";
import { utterancesOf, checkSymbols, limited, wav } from "./text.js";

const lug = createRequire(import.meta.url)("../web/public/lug.js");
const HUB = "https://huggingface.co/henryhale/tiny-voca-lg/resolve/main/gguf/";

const options = { voice: "M2", output: "tiny-voca-lg.wav", type: "q8_0", file: "" };
const words = [];
for (let args = process.argv.slice(2); args.length; ) {
  const arg = args.shift();
  if (arg.startsWith("--") && arg.slice(2) in options) options[arg.slice(2)] = args.shift();
  else words.push(arg);
}
const text = words.join(" ");
if (!text) {
  console.log('node js/say.mjs "Oli otya?" [--voice F2|M2] [--output out.wav] [--type f32|f16|q8_0|q4_0] [--file path-or-url]');
  process.exit(1);
}

const source = options.file || `${HUB}tiny-voca-lg-${options.type}.gguf`;
const started = performance.now();
const buffer = /^https?:/.test(source) ? await (await fetch(source)).arrayBuffer() : new Uint8Array(readFileSync(source)).buffer;
const fetched = performance.now();
const voca = new TinyVoca(readGGUF(buffer));
checkSymbols(voca, lug);
const ready = performance.now();

const utterances = utterancesOf(text, lug, voca.config.max_tokens - 12);
if (!utterances.length) {
  console.log("tiny-voca-lg only reads Luganda written in Latin letters.");
  process.exit(1);
}
const chunks = utterances.map(({ ids, stress }, seed) => voca.synthesize(ids, stress, { voice: options.voice, seed }).audio);
const whole = new Float32Array(chunks.reduce((sum, chunk) => sum + chunk.length, 0));
chunks.reduce((at, chunk) => (whole.set(chunk, at), at + chunk.length), 0);
const done = performance.now();
writeFileSync(options.output, new Uint8Array(wav(limited(whole), voca.config.sample_rate)));

const seconds = whole.length / voca.config.sample_rate, work = (done - ready) / 1000;
console.log(`${options.output}: ${seconds.toFixed(2)} s of audio in ${work.toFixed(2)} s, ${(seconds / work).toFixed(1)} times real time`);
console.log(`file read in ${(fetched - started).toFixed(0)} ms, model ready in ${(ready - fetched).toFixed(0)} ms`);
