#!/usr/bin/env node
// Extract turn delimiters and stop-sequence candidates from a model's chat template.
// Use this when adding a catalog entry or when a model changes.
//
//   node scripts/inspect-chat-template.mjs onnx-community/gemma-4-E2B-it-ONNX
//
// Review the output by hand before writing it into src/catalog.ts. This is not a
// generator — templates are Jinja with conditional branches, and deciding which
// branch is the default path requires judgement.

const repo = process.argv[2];
if (!repo) {
  console.error('usage: node scripts/inspect-chat-template.mjs <hf-repo-id>');
  process.exit(1);
}

const base = `https://huggingface.co/${repo}/raw/main`;

async function getJson(path) {
  const res = await fetch(`${base}/${path}`);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.json();
}

const tk = await getJson('tokenizer_config.json');

console.log(`=== ${repo} ===\n`);

console.log('eos_token :', JSON.stringify(tk.eos_token));
console.log('bos_token :', JSON.stringify(tk.bos_token));
console.log('pad_token :', JSON.stringify(tk.pad_token));

const tmpl = typeof tk.chat_template === 'string'
  ? tk.chat_template
  : tk.chat_template?.[0]?.template;

if (!tmpl) {
  console.log('\nNo chat_template found — check the model card or the source repo.');
  process.exit(0);
}

// Special-token candidates. Gemma 4 uses a <|turn> / <turn|> pair while
// Gemma 2/3 use <start_of_turn> / <end_of_turn> — the shapes differ, so match both.
const tokens = [...new Set(tmpl.match(/<\|[^|>]*\|?>|<\/?[a-z_]+\|>|<[a-z_]+>/gi) ?? [])];
console.log('\n--- special tokens appearing in the template ---');
tokens.forEach((t) => console.log(' ', JSON.stringify(t)));

// The model's turn-start string: literals emitted by the add_generation_prompt block.
const genIdx = tmpl.lastIndexOf('add_generation_prompt');
if (genIdx !== -1) {
  const tail = tmpl.slice(genIdx, genIdx + 400);
  const literals = [...tail.matchAll(/'([^']*)'/g)].map((m) => m[1]).filter(Boolean);
  console.log('\n--- generation prompt literals (turn-start candidates) ---');
  literals.forEach((l) => console.log(' ', JSON.stringify(l)));
}

// Turn-end candidates: closing tokens mirroring an opening one.
const closing = tokens.filter((t) => /^<[a-z_]+\|>$/i.test(t));
console.log('\n--- turn-end candidates (use as stop sequences) ---');
closing.forEach((t) => console.log(' ', JSON.stringify(t)));

console.log(`
Next steps:
  1. Write a src/catalog.ts entry from the turn-start and turn-end strings above.
  2. The MediaPipe path must truncate on the stop sequence itself — MediaPipe's
     built-in stop tokens are hardcoded to <eos> and <|endoftext|>.
  3. Generate something and confirm the answer does not repeat. Repetition means
     the turn-end string is wrong.
`);
