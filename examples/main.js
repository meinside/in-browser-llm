import {
  probe,
  inspectModel,
  gate,
  load,
  createMetrics,
  requestPersistence,
  parseToolCalls,
  lookupCatalog,
} from '../dist/index.js';

const $ = (sel) => document.querySelector(sel);
const mib = (n) => `${(n / 1024 / 1024).toFixed(0)}MiB`;

const REFS = {
  // Loads quickly, tokenizes non-Latin scripts efficiently (16 tokens on a sentence
  // where granite-350m needs 35, 21 of them broken UTF-8 fragments), and calls
  // tools. Sidecar weights, so it sidesteps the wasm-heap limit that blocks
  // embedded exports of this size.
  small: {
    runtime: 'transformers',
    repo: 'onnx-community/Qwen3.5-0.8B-Text-ONNX',
    // This export's generation_config.json carries only eos_token_id, so
    // transformers.js falls back to do_sample: false and the model repeats one
    // phrase until the token cap. These are Qwen's own recommended values for
    // non-thinking text tasks, from the Qwen3.5-0.8B card. top_k is included and
    // presence_penalty is not, because transformers.js implements the former and has
    // no equivalent for the latter.
    overrides: {
      do_sample: true,
      temperature: 1.0,
      top_p: 1.0,
      top_k: 20,
    },
  },
  mediapipe: {
    runtime: 'mediapipe',
    url: 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.task',
  },
  // Kept as a NEGATIVE case, deliberately. The library's options work — MediaPipe
  // builds the graph and reaches LlmVisionInferenceCalculator — but both web
  // containers ship a text decoder only, so it fails with "Image models could not
  // be created." Removing this entry would make the failure look like ours the next
  // time someone tries. See CLAUDE.md trap 13.
  //
  // One entry per modality, same URL, so the second is a cache hit. Both listed because
  // the two calculators are separate code paths — one passing says nothing about the
  // other. There is deliberately no image+audio entry: it was tried, to reach the
  // load-time refusal, and load succeeded with the failure arriving from the calculator
  // as usual. An entry that duplicates the generate-time failure under a label promising
  // load is worse than none.
  'mediapipe-mm': {
    runtime: 'mediapipe',
    url: 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm',
    modalities: { image: true },
  },
  'mediapipe-audio': {
    runtime: 'mediapipe',
    url: 'https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm/resolve/main/gemma-4-E2B-it-web.litertlm',
    modalities: { audio: true },
  },
  transformers: {
    runtime: 'transformers',
    repo: 'onnx-community/gemma-4-E2B-it-ONNX',
  },
  // The first vision model whose tensors fit Firefox: largest is 148MiB against
  // a 1024MiB limit, where gemma-4's is 1120MiB.
  vision: {
    runtime: 'transformers',
    repo: 'onnx-community/Qwen3-VL-2B-Instruct-ONNX',
  },
};

let caps = null;
let ref = null;
let engine = null;
let spec = null;
let imageUrl = null;
let audioBlob = null;
let audioUrl = null;

// `modalities` and `overrides` are fields of this page's table, not of ModelRef —
// split them off so the library never receives a field it does not define.
let asked = { image: false, audio: false };
let sampling = undefined;

$('#check').addEventListener('click', async () => {
  const { modalities = {}, overrides, ...selected } = REFS[$('#runtime').value];
  ref = selected;
  asked = { image: modalities.image ?? false, audio: modalities.audio ?? false };
  sampling = overrides;
  $('#verdict').textContent = 'Checking…';

  caps = await probe();
  spec = await inspectModel(ref);
  // A .task exposes no config, so spec.vision is false for "unknown" there. Asking
  // for the modalities anyway is what surfaces the gate's warning about that.
  const wantsImage = spec.vision || asked.image;
  const wantsAudio = spec.audio || asked.audio;
  const verdict = gate(caps, spec, { wantsImage, wantsAudio });

  const lines = [
    `<p>Total download: <strong>${mib(spec.totalBytes)}</strong>` +
      `${wantsImage ? ' · images' : ''}${wantsAudio ? ' · audio' : ''}</p>`,
    spec.largestTensorBytes
      ? `<p>Largest tensor: <strong>${mib(spec.largestTensorBytes)}</strong> ` +
        `(allocation limit ${mib(caps.maxBufferSize)}, ` +
        `binding limit ${mib(caps.maxStorageBufferBindingSize)})</p>`
      : // Unknown for two different reasons, so name the right one.
        `<p>Largest tensor: unknown — ${
          spec.runtime === 'mediapipe'
            ? 'a .task file does not expose its tensors'
            : 'weights are embedded in the .onnx, so there is no external data index'
        }. The buffer-limit check is skipped.</p>`,
  ];

  if (!verdict.ok) {
    lines.push('<p class="no">Cannot run in this browser:</p><ul>');
    verdict.blockers.forEach((b) => lines.push(`<li>${b}</li>`));
    lines.push('</ul>');
    $('#verdict').innerHTML = lines.join('');
    $('#loader').hidden = true;
    return;
  }

  if (verdict.warnings.length) {
    lines.push('<ul class="warn">');
    verdict.warnings.forEach((w) => lines.push(`<li>${w}</li>`));
    lines.push('</ul>');
  }

  $('#verdict').innerHTML = lines.join('');
  $('#loader').hidden = false;
  // Offering a picker the library would reject is a worse demo than hiding it.
  $('#image-row').hidden = !wantsImage;
  $('#audio-row').hidden = !wantsAudio;
  // Vision models are no longer excluded. Their third turn used to produce nothing:
  // transformers.js mis-signs mrope position deltas on a text-only turn, so decoding
  // ran to the cap without ever reaching EOS. The adapter now corrects that delta at
  // load time. Testing it here is the point — hiding the demo would hide the
  // regression too.
  // The MediaPipe entries match the same 'gemma-4' catalog key as the ONNX one, so a
  // format is found for them too — but that path cannot do tools at all: it has no
  // chat template to render the declaration into, and no raw-token hook to read the
  // reply's special-token delimiters. Checking the runtime as well as the format is
  // what keeps a button that cannot work from appearing.
  $('#tool-row').hidden =
    ref.runtime === 'mediapipe' || lookupCatalog(ref)?.toolCallFormat === undefined;
});

$('#image').addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (imageUrl) URL.revokeObjectURL(imageUrl);
  imageUrl = file ? URL.createObjectURL(file) : null;
  $('#preview').hidden = !imageUrl;
  if (imageUrl) $('#preview').src = imageUrl;
});

$('#audio').addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  // Pass the Blob and let the library decode — that is the path worth exercising.
  audioBlob = file ?? null;
  audioUrl = file ? URL.createObjectURL(file) : null;
  $('#audio-preview').hidden = !audioUrl;
  if (audioUrl) $('#audio-preview').src = audioUrl;
});

$('#load').addEventListener('click', async () => {
  $('#load').disabled = true;
  $('#progress').hidden = false;
  const persisted = await requestPersistence();
  const started = performance.now();
  // Only the MediaPipe path reports this; transformers.js manages its own cache.
  let fromCache = null;

  try {
    engine = await load(ref, {
      capabilities: caps,
      // Decides tokenizer vs processor, so a vision model needs this.
      spec,
      // MediaPipe only, and load-time only: it deletes the model asset once
      // consumed, so setOptions() afterwards throws. 1 image because gemma-4's
      // encoder handles one per turn here. Each modality is passed separately, so the
      // two entries above reach two different failure sites.
      ...(asked.image ? { maxImages: 1 } : {}),
      ...(asked.audio ? { supportAudio: true } : {}),
      onCacheStatus: (hit) => {
        fromCache = hit;
      },
      onProgress: ({ loaded, total, ratio }) => {
        $('#progress').value = ratio;
        $('#status').textContent = total
          ? `${mib(loaded)} / ${mib(total)} (${(ratio * 100).toFixed(1)}%)`
          : `${mib(loaded)} received`;
      },
    });
    const seconds = ((performance.now() - started) / 1000).toFixed(1);
    const cache =
      fromCache === null
        ? 'cache: not reported on this runtime'
        : `cache: ${fromCache ? 'hit' : 'miss'}`;
    $('#status').textContent =
      `Ready in ${seconds}s — ${cache} — storage persisted: ${persisted}`;
    $('#chat').hidden = false;
  } catch (error) {
    $('#load').disabled = false;
    $('#status').innerHTML = `<span class="no">${error.name}: ${error.message}</span>`;
    console.error(error);
  }
});

// How to separate a model's reasoning from its answer. Both families can be split, but
// by different means, so an entry names the MECHANISM rather than carrying a delimiter:
//
//   streamEnd  — split the live deltas on this string. Qwen3.5's `</think>` is
//                special: false (id 248069), so it survives skip_special_tokens: true.
//                Its opening tag is written into the prompt, so the stream begins
//                already inside the block.
//   rawChannel — gemma-4's `<|channel>` / `<channel|>` are special: true (ids 100, 101)
//                and exist only in the text onRawText returns. That fires ONCE at
//                completion, so this split cannot be live.
//
// A model absent from this table gets no split, which is the safe default: a splitter
// that never finds its boundary would file the whole answer as reasoning.
const REASONING = {
  'Qwen3.5': { streamEnd: '</think>' },
  'gemma-4': { rawChannel: ['<|channel>', '<channel|>'] },
};

// Substring match, so 'gemma-4' hits the LiteRT URLs too — the same shared-entry trap as
// the tool format, handled by checking the runtime at the call site.
const reasoningFor = (r) =>
  Object.entries(REASONING).find(([key]) => (r.repo ?? r.url ?? '').includes(key))?.[1];

/**
 * Pulls channel blocks out of the raw text, using the clean text for the answer.
 *
 * Only the DELIMITERS are special, so the reasoning body appears in the clean stream too:
 * deleting just the delimiters from raw reproduces clean exactly, which makes the summed
 * body length the offset where the answer starts. Taking the answer from clean therefore
 * needs no regex to strip the EOS and turn framing that raw also carries.
 *
 * Verified against the real tokenizer:
 *   raw   '<|channel>thought\nlet me think\n<channel|>the answer'
 *   clean 'thought\nlet me think\nthe answer'
 */
function splitRawChannels(raw, clean, [open, close]) {
  let reasoning = '';
  let consumed = 0;
  let at = 0;

  for (;;) {
    const start = raw.indexOf(open, at);
    if (start === -1) break;
    const bodyStart = start + open.length;
    const end = raw.indexOf(close, bodyStart);
    // Unterminated: generation stopped inside the block, usually at the token cap. The
    // rest is reasoning, and there is no answer.
    if (end === -1) {
      return { reasoning: reasoning + stripChannelName(raw.slice(bodyStart)), answer: '' };
    }
    const body = raw.slice(bodyStart, end);
    reasoning += stripChannelName(body);
    consumed += body.length;
    at = end + close.length;
  }

  return { reasoning, answer: consumed === 0 ? clean : clean.slice(consumed) };
}

// A block opens with its channel name on the first line — `<|channel>thought\n…`. The
// name is routing, not reasoning, so it is not shown.
const stripChannelName = (body) => body.slice(body.indexOf('\n') + 1);

// Splits a stream that begins INSIDE a reasoning block into reasoning and answer.
//
// Accumulates instead of testing each delta because the streamer flushes on whitespace,
// so "</" and "think>" can arrive separately even though the tag is one token.
function createReasoningSplitter(close, onReasoning, onAnswer) {
  let pending = '';
  let done = false;

  const route = (delta) => {
    if (done) {
      onAnswer(delta);
      return;
    }
    pending += delta;
    const at = pending.indexOf(close);
    if (at === -1) {
      // Hold back only as much as could be a partial tag; emit the rest as reasoning.
      const safe = pending.length - (close.length - 1);
      if (safe > 0) {
        onReasoning(pending.slice(0, safe));
        pending = pending.slice(safe);
      }
      return;
    }
    onReasoning(pending.slice(0, at));
    onAnswer(pending.slice(at + close.length));
    pending = '';
    done = true;
  };

  // Required, not optional: whatever is still held back as a possible partial tag is
  // dropped otherwise. Hitting the token cap inside the block is the common case, and
  // that lost the last seven characters — invisible on a long block, which is exactly
  // why it needs saying.
  route.flush = () => {
    if (!done && pending) {
      onReasoning(pending);
      pending = '';
    }
  };
  return route;
}

$('#generate').addEventListener('click', async () => {
  $('#generate').disabled = true;
  $('#output').textContent = '';
  $('#reasoning').textContent = '';
  $('#reasoning-box').hidden = true;
  $('#metrics').textContent = 'Generating…';

  const metrics = createMetrics();
  const maxTokens = Number($('#max-tokens').value) || undefined;
  try {
    const parts = [];
    if (imageUrl) parts.push({ type: 'image', src: imageUrl });
    if (audioBlob) parts.push({ type: 'audio', src: audioBlob });
    parts.push({ type: 'text', text: $('#prompt').value });
    // A bare string when there is nothing else, to keep exercising that path.
    const content = parts.length === 1 ? $('#prompt').value : parts;

    // enable_thinking: false is the library's default, so the box being unchecked
    // passes the value that is already in effect. Sending it either way keeps the
    // checkbox honest about what it controls. Ignored by MediaPipe, which renders no
    // template, and by models whose template does not read the flag — Qwen3-VL is one.
    const thinking = $('#thinking').checked;
    const templateArgs = { enable_thinking: thinking };

    const toOutput = (text) => {
      $('#output').textContent += text;
    };
    const showReasoning = (text) => {
      $('#reasoning-box').hidden = false;
      $('#reasoning').textContent += text;
      $('#reasoning-len').textContent = $('#reasoning').textContent.length;
    };

    // Only attempt a split when thinking was requested: with the flag off Qwen's template
    // pre-fills an empty <think></think> pair in the PROMPT, so no closing tag is
    // streamed, and gemma-4 emits no channel block at all. MediaPipe is excluded because
    // it renders no template and has no raw-token hook, and it shares gemma-4's catalog
    // substring.
    const scheme = thinking && ref.runtime === 'transformers' ? reasoningFor(ref) : undefined;
    const route = scheme?.streamEnd
      ? createReasoningSplitter(scheme.streamEnd, showReasoning, toOutput)
      : toOutput;

    // The channel scheme cannot split live — onRawText fires once at the end — so the
    // merged text streams into #output and is redistributed below.
    let raw = '';

    for await (const delta of engine.generate(
      [{ role: 'user', content }],
      // overrides is undefined for models that ship their own sampling defaults, and
      // spreading undefined is a no-op — so those keep using what the model asked for.
      {
        maxTokens,
        overrides: sampling,
        templateArgs,
        ...(scheme?.rawChannel
          ? {
              onRawText: (t) => {
                raw = t;
              },
            }
          : {}),
      },
    )) {
      metrics.onChunk();
      route(delta); // deltas, so append
    }
    route.flush?.();

    if (scheme?.rawChannel && raw) {
      const split = splitRawChannels(raw, $('#output').textContent, scheme.rawChannel);
      if (split.reasoning) {
        // Rewritten rather than appended: what streamed was reasoning and answer merged,
        // so #output has to lose the part that moves.
        $('#output').textContent = split.answer;
        showReasoning(split.reasoning);
      }
    }
    const m = metrics.finish();
    $('#metrics').textContent =
      `TTFT ${m.ttftMs === null ? 'n/a' : `${m.ttftMs.toFixed(0)}ms`} · ` +
      `${m.tokensPerSec.toFixed(1)} tok/s · ${m.chunkCount} chunks` +
      // Truncation at the cap looks like a bug, so name it.
      (maxTokens ? ` · cap ${maxTokens}` : '');

    if (thinking && !scheme) {
      $('#metrics').textContent +=
        ' · reasoning cannot be separated on this model — no known boundary';
    } else if (thinking) {
      const reasoned = $('#reasoning').textContent.length > 0;
      // A reasoning block with no answer after it means the token cap landed inside the
      // block, which reads as "the model said nothing" unless named.
      if (reasoned && !$('#output').textContent.trim()) {
        $('#metrics').textContent +=
          ' · reasoning never closed — raise max tokens to see an answer';
      } else if (!reasoned) {
        $('#metrics').textContent += ' · no reasoning — this model may ignore the flag';
      }
    }
  } catch (error) {
    $('#metrics').innerHTML = `<span class="no">${error.name}: ${error.message}</span>`;
    console.error(error);
  } finally {
    $('#generate').disabled = false;
  }
});

// One tool, one hardcoded answer. A real API call would make the demo flaky, and
// what is being demonstrated is the tool round-trip rather than the weather.
//
// Two arguments of DIFFERENT types, deliberately: with one string argument all three
// encodings look alike and a regression that flattens types would pass. A boolean rather
// than a number because Number('3') coerces away harmlessly while the string 'false' is
// TRUTHY, so `if (args.celsius)` silently inverts the request — which is why the answer
// below is rendered in whichever unit was actually parsed.
const WEATHER_TOOL = {
  type: 'function',
  function: {
    name: 'get_weather',
    description: 'Get the current weather for a city.',
    parameters: {
      type: 'object',
      properties: {
        city: { type: 'string', description: 'City name' },
        celsius: {
          type: 'boolean',
          description: 'Report the temperature in Celsius instead of Fahrenheit',
        },
      },
      required: ['city', 'celsius'],
    },
  },
};

$('#tool-demo').addEventListener('click', async () => {
  const out = $('#tool-output');
  const say = (line) => {
    out.textContent += `${line}\n`;
  };
  $('#tool-demo').disabled = true;
  out.textContent = '';

  // Non-null: the button is hidden for a model without a format.
  const format = lookupCatalog(ref).toolCallFormat;

  // gemma-4's delimiters are special tokens, so they never appear in the streamed
  // deltas — parsing those would always find nothing. onRawText delivers the same
  // answer decoded with the tokens intact, which is the only text a gemma4 parser can
  // read. Requested unconditionally: for Qwen it is simply unused.
  const needsRawText = format.encoding === 'gemma4';

  // Streams into the page as it goes: if a turn hangs rather than finishing, the
  // partial output is the only evidence of where it stopped.
  const collect = async (messages) => {
    let text = '';
    let raw = '';
    let chunks = 0;
    out.textContent += '> ';
    for await (const delta of engine.generate(messages, {
      templateArgs: { tools: [WEATHER_TOOL] },
      overrides: sampling,
      onRawText: (t) => {
        raw = t;
      },
    })) {
      text += delta;
      chunks += 1;
      out.textContent += delta;
    }
    // chunks is reported rather than returned: zero chunks used to be the only visible
    // symptom of a prompt that arrived already complete, so it belongs on screen.
    out.textContent += `\n[${chunks} chunk(s), ${text.length} char(s)]\n`;
    // The displayed text stays the clean one; only parsing uses the raw form.
    return { text, forParsing: needsRawText ? raw : text };
  };

  // Names the type of each argument, since that is what separates the encodings and
  // "3" reads the same as 3 once it is in the page.
  const describe = (args) =>
    Object.entries(args)
      .map(([k, v]) => `${k}: ${JSON.stringify(v)} (${typeof v})`)
      .join(', ');

  try {
    const messages = [
      { role: 'user', content: "What's the weather in Seoul? Use Celsius." },
    ];

    say('--- turn 1: asking the model ---');
    const first = await collect(messages);

    // Names are checked against the supplied list: a near-miss like "get_wether"
    // is the likely failure at this model size.
    const calls = parseToolCalls(first.forParsing, format, ['get_weather']);
    // Continue with a stand-in when the model did not ask for the tool, so turns 2
    // and 3 stay testable on a model too small to emit a call. Otherwise a weak
    // model hides whether the round-trip itself works.
    const wanted = calls[0] ?? {
      name: 'get_weather',
      arguments: { city: 'Seoul', celsius: true },
    };
    if (calls.length === 0) {
      say(
        '\nNo usable tool call — see the console for what was discarded. Continuing ' +
          'with a stand-in call so the rest of the round-trip is still exercised.',
      );
    }
    say(`\n--- parsed: ${wanted.name} — ${describe(wanted.arguments)} ---`);
    // The xml encoding cannot do better than a string here, so say so rather than
    // letting it look like a parser bug. gemma4 and json both return a real boolean.
    if (typeof wanted.arguments.celsius === 'string') {
      say(
        `(celsius arrived as a string: the ${format.encoding} encoding carries no ` +
          'types, so a caller with a schema coerces. Note the string "false" is truthy, ' +
          'which is why this is worse than a mis-typed number.)',
      );
    }

    // The naive truthiness test a caller would write, deliberately, so a mis-typed reply
    // visibly picks the wrong unit. `unit`/`temperature` rather than reusing `celsius` for
    // the reading: it is a boolean REQUEST here, and one name meaning two things invites
    // `if (result.celsius)` — true at 21, false at 0.
    const result = JSON.stringify({
      city: wanted.arguments.city ?? 'unknown',
      ...(Boolean(wanted.arguments.celsius)
        ? { unit: 'celsius', temperature: 21 }
        : { unit: 'fahrenheit', temperature: 70 }),
      sky: 'clear',
    });
    say(`--- turn 2: our hardcoded result ---\n${result}`);

    say('\n--- turn 3: model answers with the result ---');
    const final = await collect([
      ...messages,
      { role: 'assistant', content: '', tool_calls: [wanted] },
      { role: 'tool', content: result },
    ]);
    // The prompt for this turn renders correctly (verified against the processor),
    // so silence here means the model stopped, not that a message was dropped.
    if (!final.text.trim()) {
      say('(the model produced no usable text for this turn)');
    }
  } catch (error) {
    say(`\n${error.name}: ${error.message}`);
    console.error(error);
  } finally {
    $('#tool-demo').disabled = false;
  }
});
