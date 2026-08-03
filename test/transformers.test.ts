import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  correctedMropeDeltas,
  dtypeFor,
  expectsMropePositions,
  normalizeContent,
  patchMropeDeltas,
  processorCallArgs,
  selectModelClass,
  warnIfGreedyByDefault,
} from '../src/engine/transformers.js';
import type { Capabilities } from '../src/capability.js';

// dtype selection silently changes what gets downloaded, so it is tested
// directly. A per-session dtype object once hardcoded gemma-4's session names;
// for a model shipping a single `model` session every key resolved to null and
// fp32 was fetched instead — 260MB became 1087MB with no error.
function caps(over: Partial<Capabilities> = {}): Capabilities {
  return {
    hasWebGPU: true,
    hasF16: true,
    hasSubgroups: true,
    knownF16ShaderIssues: [],
    maxStorageBufferBindingSize: 4294967292,
    maxBufferSize: 4294967292,
    adapterInfo: { vendor: 'apple', architecture: 'metal-3' },
    ...over,
  };
}

describe('dtypeFor', () => {
  it('returns a bare string, never a per-session object', () => {
    // A string applies to every session whatever the export's layout is.
    expect(typeof dtypeFor(caps())).toBe('string');
  });

  it('uses q4f16 when f16 is usable', () => {
    expect(dtypeFor(caps())).toBe('q4f16');
  });

  it('falls back to q4 when shader-f16 is absent', () => {
    expect(dtypeFor(caps({ hasF16: false }))).toBe('q4');
  });

  it('falls back to q4 when a known shader bug applies', () => {
    // Firefox 153 reports shader-f16 and compiles every canary, yet fails ORT's
    // Clip kernel at inference time.
    expect(dtypeFor(caps({ knownF16ShaderIssues: ['Firefox Clip bug'] }))).toBe('q4');
  });

  it('never selects an unquantized dtype', () => {
    // Leaving dtype unset defaults to fp32 on webgpu, which is the failure mode.
    for (const c of [caps(), caps({ hasF16: false }), caps({ knownF16ShaderIssues: ['x'] })]) {
      expect(['q4f16', 'q4']).toContain(dtypeFor(c));
    }
  });

  it('assumes f16 when capabilities were not supplied', () => {
    expect(dtypeFor(undefined)).toBe('q4f16');
  });

});

// Processor arity AND argument order are fixed per model family, not per call.
// Getting this wrong is silent-ish: gemma-4's feature extractor rejects the options
// object with "expects a Float32Array … but got Object", naming nothing about the
// real cause. These signatures are documented nowhere but the shipped
// processing_*.js sources.
describe('processorCallArgs', () => {
  const OPTIONS = { add_special_tokens: false };
  const IMAGES = ['img'];
  const AUDIO = [new Float32Array([0.5])];

  it('passes images then audio for a vision+audio model, gemma-4 order', () => {
    // processing_gemma4.js is _call(text, images, audio, options).
    expect(processorCallArgs({ vision: true, audio: true }, 'p', IMAGES, AUDIO, OPTIONS)).toEqual([
      'p',
      IMAGES,
      AUDIO,
      OPTIONS,
    ]);
  });

  it('omits the audio slot entirely for a vision-only model', () => {
    // qwen2/3-vl is _call(text, images, ...args), so a null third argument would put
    // the options object one slot too late.
    expect(processorCallArgs({ vision: true, audio: false }, 'p', IMAGES, null, OPTIONS)).toEqual([
      'p',
      IMAGES,
      OPTIONS,
    ]);
  });

  it('keeps the audio slot on a text-only turn for a model that has both', () => {
    // The trap: omitting the argument puts OPTIONS where audio belongs, and the
    // feature extractor rejects it. Arity is the family's, not the turn's.
    expect(processorCallArgs({ vision: true, audio: true }, 'p', null, null, OPTIONS)).toEqual([
      'p',
      null,
      null,
      OPTIONS,
    ]);
  });

  // UNVERIFIED shape — no audio-text model has reached inference here. Pinned to
  // UltravoxProcessor's signature so a later edit has to change it deliberately.
  it('puts one audio clip in the second slot for an audio-only model', () => {
    // _call(text, audio, kwargs): audio sits where gemma-4 puts images, and it reads
    // audio.length as a SAMPLE count, so passing the array of clips would be wrong
    // twice over.
    expect(processorCallArgs({ vision: false, audio: true }, 'p', null, AUDIO, OPTIONS)).toEqual([
      'p',
      AUDIO[0],
      OPTIONS,
    ]);
  });

  it('passes null rather than undefined when an audio-only model gets no audio', () => {
    // undefined would be dropped by a defaulted parameter; null is explicit.
    expect(processorCallArgs({ vision: false, audio: true }, 'p', null, null, OPTIONS)).toEqual([
      'p',
      null,
      OPTIONS,
    ]);
  });
});

describe('selectModelClass', () => {
  const CLASSES = {
    causalLM: 'causalLM',
    imageTextToText: 'imageTextToText',
    audioTextToText: 'audioTextToText',
  };

  it('keeps a vision+audio model on the image-text-to-text class', () => {
    // gemma-4 declares both and is registered ONLY under ImageTextToText, so the
    // vision test must come first. Reversing it throws "Unsupported model type".
    expect(selectModelClass(CLASSES, { vision: true, audio: true })).toBe('imageTextToText');
  });

  it('uses the image class for a vision-only model', () => {
    expect(selectModelClass(CLASSES, { vision: true, audio: false })).toBe('imageTextToText');
  });

  it('uses the audio class for an audio-only model', () => {
    // ultravox is absent from the CausalLM and ImageTextToText mappings entirely.
    expect(selectModelClass(CLASSES, { vision: false, audio: true })).toBe('audioTextToText');
  });

  it('uses the causal-LM class for a text-only model', () => {
    expect(selectModelClass(CLASSES, { vision: false, audio: false })).toBe('causalLM');
  });
});

// Opposite requirements per family. Vision templates read
// message['content'][0]['type'] and loop over it, so a string fails with "Expected
// iterable or object type in for loop: got StringValue". Ultravox's Llama-derived
// template applies `content | trim`, which fails on an array with "Unknown ArrayValue
// filter: trim" — observed in Firefox before this branch existed.
describe('normalizeContent', () => {
  it('wraps a string as parts for a vision model', () => {
    expect(normalizeContent([{ role: 'user', content: 'hi' }], true)).toEqual([
      { role: 'user', content: [{ type: 'text', text: 'hi' }] },
    ]);
  });

  it('flattens parts to a string for a non-vision model', () => {
    expect(
      normalizeContent(
        [
          {
            role: 'user',
            content: [
              { type: 'audio', data: new Float32Array(1) },
              { type: 'text', text: 'hi' },
            ],
          },
        ],
        false,
      ),
    ).toEqual([{ role: 'user', content: 'hi' }]);
  });

  // The turn-3 bug: an assistant turn carrying only tool_calls has content '', and
  // wrapping that as [{type:'text',text:''}] flips it from falsy to truthy. gemma-4's
  // template branches on exactly that — `not message.get('content')` — and closed the
  // model's turn, after which add_generation_prompt declines to reopen it following a
  // tool_response. The model received a finished conversation and returned EOS: zero
  // chunks, no error, nothing naming the cause.
  it('leaves empty content empty rather than wrapping it in a part', () => {
    expect(normalizeContent([{ role: 'assistant', content: '' }], true)).toEqual([
      { role: 'assistant', content: '' },
    ]);
  });

  it('leaves empty content alone on the non-vision path too', () => {
    // toText('') already returns '', so this direction was never broken — pinned so a
    // future rewrite cannot make the two paths disagree again.
    expect(normalizeContent([{ role: 'assistant', content: '' }], false)).toEqual([
      { role: 'assistant', content: '' },
    ]);
  });

  it('still wraps a non-empty string for a vision model', () => {
    // The empty-content guard must not disable the wrapping the vision path needs.
    expect(normalizeContent([{ role: 'user', content: 'hi' }], true)).toEqual([
      { role: 'user', content: [{ type: 'text', text: 'hi' }] },
    ]);
  });

  it('wraps an empty text part left explicitly by the caller', () => {
    // Already an array, so nothing is being converted and the falsiness question does
    // not arise — a caller who built parts by hand gets them through untouched.
    const content = [{ type: 'text' as const, text: '' }];
    expect(normalizeContent([{ role: 'assistant', content }], true)).toEqual([
      { role: 'assistant', content },
    ]);
  });

  it('preserves other message fields, tool_calls among them', () => {
    // An assistant turn replays its own call; dropping it breaks the next turn. By
    // this point toTemplateToolCalls has already nested them. Content stays '' — see
    // the empty-content tests above for why that matters.
    const tool_calls = [
      { type: 'function' as const, function: { name: 'get_weather', arguments: { city: 'Seoul' } } },
    ];
    expect(normalizeContent([{ role: 'assistant', content: '', tool_calls }], true)).toEqual([
      { role: 'assistant', content: '', tool_calls },
    ]);
  });

  it('keeps a non-empty vision turn wrapped even when it carries tool calls', () => {
    // The guard is about emptiness, not about tool calls — a turn with both must still
    // get the parts a vision template needs.
    const tool_calls = [
      { type: 'function' as const, function: { name: 'f', arguments: {} } },
    ];
    expect(normalizeContent([{ role: 'assistant', content: 'ok', tool_calls }], true)).toEqual([
      { role: 'assistant', content: [{ type: 'text', text: 'ok' }], tool_calls },
    ]);
  });
});

// Works around a sign error in transformers.js 4.2.0: get_rope_index's text-only
// branch computes mrope_position_deltas as max(positions) + 1 + seq_len
// (modeling_qwen2_vl.js:37) where its multimodal branch uses ... - seq_len
// (line 257). The value is consumed as an offset — each decode step is placed at
// past_length + delta — so for a contiguous sequence it must be 0. Upstream yields
// 2*seq_len, putting generated tokens near 3*seq_len, outside the trained range:
// output collapses to one repeated token and EOS never arrives.
describe('correctedMropeDeltas', () => {
  // position_ids are [3, batch, seq] with the three mrope sections identical for text.
  const positions = (batch: number, seq: number) => ({
    dims: [3, batch, seq],
    data: BigInt64Array.from({ length: 3 * batch * seq }, (_, i) => BigInt(i % seq)),
  });

  it('is zero for a contiguous sequence, where upstream gives 2*seq_len', () => {
    // The whole bug in one assertion.
    for (const seq of [1, 10, 160]) {
      expect(Array.from(correctedMropeDeltas(positions(1, seq)))).toEqual([0n]);
    }
  });

  it('matches the sign the multimodal branch uses', () => {
    // max(positions) + 1 - seq_len. With vision segments max exceeds seq_len - 1, so
    // the delta is positive — that case must keep working, not just the zero one.
    const withVisionSpread = {
      dims: [3, 1, 4],
      data: BigInt64Array.from([0n, 1n, 5n, 9n, 0n, 1n, 5n, 9n, 0n, 1n, 5n, 9n]),
    };
    expect(Array.from(correctedMropeDeltas(withVisionSpread))).toEqual([10n - 4n]);
  });

  it('computes one delta per batch element', () => {
    const deltas = correctedMropeDeltas(positions(3, 8));
    expect(deltas.length).toBe(3);
    expect(Array.from(deltas)).toEqual([0n, 0n, 0n]);
  });

  it('reads only the first mrope section, not across section boundaries', () => {
    // Sections are contiguous in memory. Reading past one would pick up another
    // section's values and inflate max, so the delta must come from section 0 alone.
    const seq = 5;
    const p = positions(1, seq);
    expect(Array.from(correctedMropeDeltas(p))).toEqual([0n]);
  });
});

describe('patchMropeDeltas', () => {
  class FakeTensor {
    constructor(
      public type: string,
      public data: BigInt64Array,
      public dims: number[],
    ) {}
  }

  // Mimics upstream: correct position_ids, delta with the wrong sign.
  const makeModel = (model_type: string, seq = 6) => ({
    config: { model_type },
    calls: 0,
    _get_text_only_rope_index(this: { calls: number }) {
      this.calls += 1;
      const data = BigInt64Array.from({ length: 3 * seq }, (_, i) => BigInt(i % seq));
      return [
        { data, dims: [3, 1, seq] },
        // The bug: max + 1 + seq instead of max + 1 - seq.
        { data: BigInt64Array.from([BigInt(seq - 1) + 1n + BigInt(seq)]), dims: [1, 1] },
      ];
    },
  });

  it('replaces the delta while leaving position_ids alone', () => {
    const model = makeModel('qwen3_vl');
    patchMropeDeltas(model, FakeTensor as never);
    const [positionIds, deltas] = model._get_text_only_rope_index() as never as [
      { data: BigInt64Array },
      { data: BigInt64Array },
    ];
    // Positions were never wrong, so they must be untouched.
    expect(Array.from(positionIds.data.subarray(0, 6))).toEqual([0n, 1n, 2n, 3n, 4n, 5n]);
    expect(Array.from(deltas.data)).toEqual([0n]);
  });

  it('keeps working across repeated calls, which is the point', () => {
    // _update_model_kwargs_for_generation nulls position_ids after every step
    // (modeling_utils.js:687), so this runs once per generated token. An earlier fix
    // that supplied position_ids to generate() corrected only the first one.
    const model = makeModel('qwen3_vl');
    patchMropeDeltas(model, FakeTensor as never);
    for (let step = 0; step < 4; step++) {
      const [, deltas] = model._get_text_only_rope_index() as never as [
        unknown,
        { data: BigInt64Array },
      ];
      expect(Array.from(deltas.data)).toEqual([0n]);
    }
    expect(model.calls).toBe(4);
  });

  it('leaves models without mrope untouched', () => {
    // gemma-4 is a vision model that does not use mrope; patching it would break a
    // working path.
    const model = makeModel('gemma4');
    patchMropeDeltas(model, FakeTensor as never);
    const [, deltas] = model._get_text_only_rope_index() as never as [
      unknown,
      { data: BigInt64Array },
    ];
    // Upstream's buggy value, left in place: (seq-1) + 1 + seq = 2*seq = 12.
    expect(Array.from(deltas.data)).toEqual([12n]);
  });

  it('is idempotent, so a second call does not double-wrap', () => {
    const model = makeModel('qwen2_5_vl');
    patchMropeDeltas(model, FakeTensor as never);
    patchMropeDeltas(model, FakeTensor as never);
    const [, deltas] = model._get_text_only_rope_index() as never as [
      unknown,
      { data: BigInt64Array },
    ];
    expect(Array.from(deltas.data)).toEqual([0n]);
  });

  it('does nothing when the method is absent', () => {
    // A future version may rename or remove it; that must not throw at load time.
    const model = { config: { model_type: 'qwen3_vl' } };
    expect(() => patchMropeDeltas(model, FakeTensor as never)).not.toThrow();
  });
});

describe('expectsMropePositions', () => {
  it('matches the qwen-vl families that carry the bug', () => {
    for (const model_type of ['qwen2_vl', 'qwen2_5_vl', 'qwen3_vl']) {
      expect(expectsMropePositions({ config: { model_type } })).toBe(true);
    }
  });

  it('leaves other models alone', () => {
    // gemma-4 is a vision model too but does not use mrope, so overriding its
    // position_ids would break a path that works.
    expect(expectsMropePositions({ config: { model_type: 'gemma4' } })).toBe(false);
    expect(expectsMropePositions({ config: { model_type: 'qwen3' } })).toBe(false);
  });

  it('does not throw on a model without a config', () => {
    expect(expectsMropePositions({})).toBe(false);
    expect(expectsMropePositions(undefined)).toBe(false);
  });
});

// transformers.js defaults do_sample to false (configuration_utils.js:68), so a model
// shipping no sampling parameters decodes greedily. Observed on Qwen3.5-0.8B-Text-ONNX,
// whose generation_config.json holds only eos_token_id: a conversational prompt came
// back as one phrase repeated to the token cap. The other two models in the lineup do
// ship do_sample: true, which is why only this one showed it.
describe('warnIfGreedyByDefault', () => {
  const captureWarning = (model: unknown) => {
    const seen: string[] = [];
    const original = console.warn;
    console.warn = (msg: string) => seen.push(msg);
    try {
      warnIfGreedyByDefault(model, 'some/model');
    } finally {
      console.warn = original;
    }
    return seen;
  };

  it('warns when the config carries no sampling parameters at all', () => {
    const warnings = captureWarning({ generation_config: { eos_token_id: 248044 } });
    expect(warnings).toHaveLength(1);
    // Naming the cause is the whole point; "repeats" is the symptom users see.
    expect(warnings[0]).toMatch(/greedily/);
    expect(warnings[0]).toMatch(/overrides/);
  });

  it('stays quiet when the model asks for sampling', () => {
    expect(captureWarning({ generation_config: { do_sample: true, top_k: 20 } })).toEqual([]);
  });

  it('stays quiet when the model deliberately asks for greedy', () => {
    // do_sample: false present is a choice by the model author; absent is the silent
    // case this warning is about. Warning on both would train people to ignore it.
    expect(captureWarning({ generation_config: { do_sample: false } })).toEqual([]);
  });

  it('treats temperature or top_k alone as an intentional configuration', () => {
    expect(captureWarning({ generation_config: { temperature: 0.7 } })).toEqual([]);
    expect(captureWarning({ generation_config: { top_k: 50 } })).toEqual([]);
  });

  it('says nothing when there is no generation config to judge', () => {
    // Absent config is not evidence of greedy decoding, so guessing would be noise.
    expect(captureWarning({ generation_config: null })).toEqual([]);
    expect(captureWarning({})).toEqual([]);
  });
});

// Two assertions about the source text rather than about behaviour, because what they
// check is WIRING: whether the patch is installed at all, and whether an approach that
// was tried and failed has crept back. Neither is observable without a WebGPU adapter
// and multi-GB weights, so a unit test cannot reach them. The tests above cover what
// patchMropeDeltas does once installed.
describe('where the mrope fix is applied', () => {
  const source = readFileSync(
    fileURLToPath(new URL('../src/engine/transformers.ts', import.meta.url)),
    'utf8',
  );

  it('patches at load time rather than per generate call', () => {
    // Per-call is not enough: the delta is recomputed for every token, so a
    // position_ids override fixes only the first one.
    expect(source).toMatch(/patchMropeDeltas\(model, Tensor\)/);
  });

  it('does not pass position_ids into generate', () => {
    // The approach that did not work. If this reappears, the repetition bug is back.
    expect(source).not.toMatch(/position_ids:/);
  });
});
