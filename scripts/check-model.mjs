#!/usr/bin/env node
// Pre-flight checks for adding or changing a supported model.
//
//   node scripts/check-model.mjs onnx-community/gemma-3-270m-it-ONNX
//
// Every check here exists because it failed silently at least once. Downloads
// only a few KB — headers and JSON configs, never weights.

const repo = process.argv[2];
if (!repo) {
  console.error('usage: node scripts/check-model.mjs <hf-repo-id>');
  process.exit(1);
}

const BASE = `https://huggingface.co/${repo}`;
const problems = [];
const notes = [];

async function json(path) {
  const res = await fetch(`${BASE}/raw/main/${path}`);
  return res.ok ? res.json() : null;
}

async function head(path) {
  const res = await fetch(`${BASE}/resolve/main/${path}`, { method: 'HEAD' });
  return res.ok ? Number(res.headers.get('content-length')) || 0 : 0;
}

const mib = (n) => `${(n / 1024 / 1024).toFixed(0)}MiB`;

// Counts an ASCII needle in raw bytes, avoiding a full string decode — inline
// weights make these files far too large to stringify.
function countAscii(bytes, needle) {
  const pattern = [...needle].map((c) => c.charCodeAt(0));
  let count = 0;
  outer: for (let i = 0; i <= bytes.length - pattern.length; i += 1) {
    for (let j = 0; j < pattern.length; j += 1) {
      if (bytes[i + j] !== pattern[j]) continue outer;
    }
    count += 1;
  }
  return count;
}

console.log(`=== ${repo} ===\n`);

// 1. Gating. A gated repo cannot be fetched from a browser at all.
const info = await (await fetch(`https://huggingface.co/api/models/${repo}`)).json();
if (info.gated) problems.push(`Repo is gated (${info.gated}); browsers cannot fetch it.`);
console.log(`gated       : ${info.gated ?? false}`);
console.log(`license     : ${info.cardData?.license ?? 'unknown'}`);

// 2. Session layout. inspect.ts probes a fixed list of names; an export using
// something else reports every size as 0, which the gate reads as "fits" and so
// silently disables the largest-tensor check.
const KNOWN_SESSIONS = [
  'model',
  'embed_tokens',
  'decoder_model_merged',
  'vision_encoder',
  'audio_encoder',
];
const onnxFiles = (info.siblings ?? [])
  .map((s) => s.rfilename)
  .filter((f) => f.startsWith('onnx/') && f.endsWith('.onnx'));

const sessions = new Set(
  onnxFiles.map((f) =>
    f
      .replace(/^onnx\//, '')
      .replace(/\.onnx$/, '')
      .replace(/_(q4f16|q4|fp16|quantized|q8|bnb4|int8|uint8)$/, ''),
  ),
);
console.log(`\nsessions    : ${[...sessions].join(', ') || '(none found)'}`);
for (const s of sessions) {
  if (!KNOWN_SESSIONS.includes(s)) {
    problems.push(
      `Session "${s}" is not in inspect.ts CANDIDATE_SESSIONS — its size would ` +
        'read as 0 and the tensor-limit gate would be skipped. Add it.',
    );
  }
}

// 3. Quantized variants. Without one, the browser downloads fp32.
const variants = ['q4f16', 'q4', 'fp16', 'quantized'];
// Weights live either in a sidecar .onnx_data or inline in the .onnx, depending
// on whether the export exceeded protobuf's 2GB limit. Count both.
console.log('\nvariant sizes:');
const sizes = {};
// Per variant, the largest .onnx with no sidecar. ORT mallocs such a file whole,
// so it is a separate constraint from the total — tracked per session, not summed.
const largestEmbedded = {};
for (const v of variants) {
  let total = 0;
  let embedded = 0;
  for (const s of sessions) {
    const sidecar = await head(`onnx/${s}_${v}.onnx_data`);
    if (sidecar > 0) {
      total += sidecar;
    } else {
      const inline = await head(`onnx/${s}_${v}.onnx`);
      total += inline;
      embedded = Math.max(embedded, inline);
    }
  }
  if (total > 0) sizes[v] = total;
  largestEmbedded[v] = embedded;
  const note = embedded > 0 ? ` (largest embedded .onnx ${mib(embedded)})` : '';
  console.log(`  ${v.padEnd(10)} ${total > 0 ? mib(total) : '—'}${note}`);
}

// 2GiB is documented (ONNX protobuf, Chrome ArrayBuffer). 1GiB is arbitrary: a
// 1360MiB embedded file threw std::bad_alloc on Firefox while 949MiB of sidecar
// data loaded, and no source names a limit in between.
const worstEmbedded = Math.max(...Object.values(largestEmbedded), 0);
if (worstEmbedded >= 2 * 1024 * 1024 * 1024) {
  notes.push(
    `MUST FIX: a single .onnx is ${mib(worstEmbedded)} with embedded weights, at or over ` +
      'the 2GB ONNX protobuf limit. ONNX Runtime cannot load it in a browser.',
  );
} else if (worstEmbedded > 1024 * 1024 * 1024) {
  notes.push(
    `CHECK: a single .onnx is ${mib(worstEmbedded)} with embedded weights, which ORT must ` +
      'allocate contiguously in a 4GiB wasm heap and duplicate while parsing. 1360MiB ' +
      'failed with std::bad_alloc on Firefox. Prefer an export using .onnx_data.',
  );
}
if (!sizes.q4f16 && !sizes.q4) {
  problems.push('Neither q4f16 nor q4 exists; loading would fall back to fp32.');
}

// 4. Clip nodes. These are fp16 overflow guards that Firefox fails to compile
// ("ShaderModule with 'Clip' label is invalid"). Their presence means the model
// must run q4 on Firefox.
const clipCounts = {};
console.log('\nf16 Clip guards (fail to compile on Firefox):');
for (const v of ['q4f16', 'q4']) {
  for (const s of sessions) {
    // Read only the graph header. When weights are inline the file can be
    // gigabytes, and decoding that to a string overflows V8's string limit.
    const res = await fetch(`${BASE}/resolve/main/onnx/${s}_${v}.onnx`, {
      headers: { Range: 'bytes=0-8388607' },
    });
    if (!res.ok) continue;
    const bytes = new Uint8Array(await res.arrayBuffer());
    const count = countAscii(bytes, 'Clip');
    console.log(`  ${s}_${v}`.padEnd(34) + `${count} Clip`);
    clipCounts[`${s}_${v}`] = count;
  }
}

for (const s of sessions) {
  const f16 = clipCounts[`${s}_q4f16`] ?? 0;
  const q4 = clipCounts[`${s}_q4`] ?? 0;
  if (f16 > 0 && q4 === 0) {
    notes.push(
      `${s}: q4f16 has ${f16} Clip nodes but q4 has none, so downgrading dtype ` +
        'avoids the Firefox failure. Confirm knownF16ShaderIssues still triggers.',
    );
  } else if (q4 > 0) {
    // Encoders keep their Clip guards even at q4, so the dtype downgrade cannot
    // rescue them — this is why gemma-4 stays Firefox-blocked for other reasons too.
    notes.push(
      `${s}: q4 still has ${q4} Clip nodes, so a dtype downgrade will NOT avoid ` +
        'the Firefox shader failure for this session. Expect it to fail there.',
    );
  }
}

// 5. Sampling and stop tokens. Overriding these caused rambling output once.
const gen = await json('generation_config.json');
console.log('\ngeneration_config:');
if (!gen) {
  notes.push('No generation_config.json; sampling falls back to library defaults.');
  console.log('  (absent)');
} else {
  console.log(`  do_sample     : ${gen.do_sample ?? '(unset)'}`);
  console.log(`  top_k / top_p : ${gen.top_k ?? '—'} / ${gen.top_p ?? '—'}`);
  console.log(`  eos_token_id  : ${JSON.stringify(gen.eos_token_id)}`);
  if (Array.isArray(gen.eos_token_id)) {
    notes.push(
      `eos_token_id is an array (${JSON.stringify(gen.eos_token_id)}); ` +
        'transformers.js handles that — do not reimplement stop detection.',
    );
  }
  // A config that specifies NO sampling is worse than one specifying greedy: the
  // library's default is do_sample false, so the model repeats itself and nothing in
  // the file says why. Mistaken for "the model is too small" on Qwen3.5-0.8B.
  const specifiesSampling =
    'do_sample' in gen || 'temperature' in gen || 'top_k' in gen || 'top_p' in gen;
  if (!specifiesSampling) {
    notes.push(
      'generation_config.json specifies no sampling parameters, so transformers.js ' +
        'decodes GREEDILY (its do_sample default is false). Small models repeat one ' +
        'phrase to the token cap that way, which looks like a weak model rather than ' +
        "a missing default. Read the source model's card and pass its recommended " +
        'values via GenerateOptions.overrides.',
    );
  }
}

// 6. Turn format. Only the MediaPipe path needs a catalog entry, because it has
// no chat template support; transformers.js applies the Jinja template itself.
const tk = await json('tokenizer_config.json');
const template = typeof tk?.chat_template === 'string' ? tk.chat_template : tk?.chat_template?.[0]?.template;
console.log('\nturn format:');
if (!template) {
  notes.push('No chat_template found; verify the prompt format by hand.');
  console.log('  (no chat_template)');
} else {
  const shapes = [
    ['<|turn>', 'gemma-4 style'],
    ['<start_of_turn>', 'gemma-2/3 style'],
    ['<|im_start|>', 'ChatML style'],
  ];
  const found = shapes.filter(([tok]) => template.includes(tok));
  console.log(`  ${found.map(([t, d]) => `${t} (${d})`).join(', ') || 'unrecognized'}`);
  notes.push(
    'For the MediaPipe path only, add a catalog entry — run ' +
      'scripts/inspect-chat-template.mjs. transformers.js needs none.',
  );
}

// Which adapter path a model needs. Detection keys off the encoder sub-configs
// rather than the architecture name, which differs per family.
const modelConfig = await json('config.json');
console.log('\nmodalities:');
if (!modelConfig) {
  console.log('  (no config.json — treated as text-only)');
} else {
  const vision = !!modelConfig.vision_config;
  const audio = !!modelConfig.audio_config;
  console.log(`  architectures : ${JSON.stringify(modelConfig.architectures)}`);
  console.log(`  vision        : ${vision}`);
  console.log(`  audio         : ${audio}`);
  if (vision) {
    notes.push('Vision model: pass the inspectModel() result as LoadOptions.spec when loading.');
  }
  if (audio) {
    notes.push('Audio model: pass the inspectModel() result as LoadOptions.spec so the processor receives its audio argument.');
  }

  // AutoProcessor trusts this field unconditionally, so a wrong value means the
  // model's own processor is never constructed. Ultravox names WhisperProcessor,
  // whose _call takes audio FIRST, and the prompt string ends up in the audio slot.
  if (vision || audio) {
    const pre = (await json('preprocessor_config.json')) ?? (await json('processor_config.json'));
    const declared = pre?.processor_class;
    console.log(`  processor     : ${declared ?? '(not declared)'}`);
    if (declared && /^(Whisper|Wav2Vec2|CLIP|Siglip)Processor$/.test(declared)) {
      notes.push(
        `MUST FIX: processor_class is "${declared}", a bare encoder processor rather than ` +
          'the model\'s own. AutoProcessor will build that one instead, and its argument ' +
          'order differs — the prompt lands in the audio/image slot. This export cannot ' +
          'work through AutoProcessor.',
      );
    }
  }
}

console.log('\n' + '='.repeat(60));
if (problems.length) {
  console.log('\nMUST FIX:');
  problems.forEach((p) => console.log(`  - ${p}`));
}
if (notes.length) {
  console.log('\nCHECK:');
  notes.forEach((n) => console.log(`  - ${n}`));
}
if (!problems.length) {
  console.log('\nNo blocking problems found.');
}
console.log(`
Still verify in a browser — none of the above proves inference works:
  1. Chrome  : gate passes, generation streams without repetition
  2. Firefox : dtype downgrades to q4, no Clip error, output is not rambling
  3. Network : the file being downloaded carries the expected dtype suffix
`);

process.exit(problems.length ? 1 : 0);
