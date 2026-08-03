import { describe, expect, it } from 'vitest';
import { explainModalityFailure } from '../src/engine/mediapipe.js';
import { CapabilityError } from '../src/errors.js';

// The verbatim error from Firefox with maxNumImages: 1 against
// gemma-4-E2B-it-web.litertlm. Kept in full because the rewrite matches on its
// shape, and a paraphrase would not prove the match works.
const VISION_GRAPH_FAILURE =
  'Failed to predict asynchronously: INVALID_ARGUMENT: CalculatorGraph::Run() ' +
  'failed: Calculator::Open() for node "LlmVisionInferenceCalculator" failed: ' +
  'Image models could not be created. ' +
  "[type.googleapis.com/mediapipe.StatusList='status { code: 3 message: " +
  '"Calculator::Open() for node \\"LlmVisionInferenceCalculator\\" failed: Image ' +
  'models could not be created." }\'] === Source Location Trace: === ' +
  'third_party/odml/infra/genai/inference/ml_drift/llm/llm_gpu_runner_manager.cc:509';

describe('explainModalityFailure', () => {
  it('replaces the graph-level vision failure with a cause that names the reason', () => {
    const rewritten = explainModalityFailure(new Error(VISION_GRAPH_FAILURE));
    expect(rewritten).toBeInstanceOf(CapabilityError);
    // The point of the rewrite: the original says "Image models could not be
    // created" and nothing about why. The replacement has to say what is missing.
    expect((rewritten as Error).message).toMatch(/no vision or audio weights/);
    expect((rewritten as Error).message).toMatch(/text decoder only/);
  });

  it('keeps the original reachable as cause', () => {
    // Rewriting an error discards evidence unless the original is preserved.
    const original = new Error(VISION_GRAPH_FAILURE);
    const rewritten = explainModalityFailure(original) as Error;
    expect(rewritten.cause).toBe(original);
  });

  // supportAudio reaches this at predict time exactly as maxImages does, alone or
  // together. `LlmAudioInferenceCalculator` is present in genai_wasm_internal.wasm, so
  // this is a real path rather than a symmetric guess — an earlier note here claimed audio
  // could only fail at load.
  it('matches the audio calculator too', () => {
    const rewritten = explainModalityFailure(
      new Error(
        'Calculator::Open() for node "LlmAudioInferenceCalculator" failed: Audio ' +
          'models could not be created.',
      ),
    );
    expect(rewritten).toBeInstanceOf(CapabilityError);
  });

  // Verbatim from one Firefox run, and NOT reproducible since: neither modality alone
  // nor both together get here now — load succeeds and the calculator fails instead. The
  // trigger is unknown, so this documents a message rather than a condition. Kept
  // because it was really seen, it surfaces from createFromOptions rather than the graph,
  // and it shares none of the predict-time wording, so nothing else here would catch it.
  it('matches the load-time audio refusal, which names no calculator', () => {
    const loadTime =
      'Failed to create session: INVALID_ARGUMENT: Audio options should not be null ' +
      'when loading audio runner. === Source Location Trace: === ' +
      'third_party/odml/infra/genai/inference/ml_drift/llm/llm_gpu_runner_manager.cc:572';
    const rewritten = explainModalityFailure(new Error(loadTime));
    expect(rewritten).toBeInstanceOf(CapabilityError);
    expect((rewritten as Error).message).toMatch(/no vision or audio weights/);
  });

  it('matches the vision equivalent of that load-time refusal', () => {
    const rewritten = explainModalityFailure(
      new Error('INVALID_ARGUMENT: Vision options should not be null.'),
    );
    expect(rewritten).toBeInstanceOf(CapabilityError);
  });

  it('matches the underlying reason if it ever surfaces directly', () => {
    // The wasm's own diagnostic, and the clearest statement of the cause:
    // "Audio is enabled but audio_encoder_parameters are not defined in the model".
    const rewritten = explainModalityFailure(
      new Error('Audio is enabled but audio_encoder_parameters are not defined in the model'),
    );
    expect(rewritten).toBeInstanceOf(CapabilityError);
  });

  it('passes an unrelated INVALID_ARGUMENT through untouched', () => {
    // Both halves of the match are required, so a different graph failure keeps its
    // own message rather than being mislabeled a modality problem.
    const other = new Error('INVALID_ARGUMENT: CalculatorGraph::Run() failed: something else');
    expect(explainModalityFailure(other)).toBe(other);
  });

  it('passes a merely similar message through untouched', () => {
    // Names the calculator but did not fail to create a model.
    const other = new Error('LlmVisionInferenceCalculator timed out');
    expect(explainModalityFailure(other)).toBe(other);
  });

  it('does not claim a missing-modality cause for an unrelated null-option error', () => {
    // "should not be null" alone is too generic to attribute to absent weights.
    const other = new Error('INVALID_ARGUMENT: Sampler options should not be null.');
    expect(explainModalityFailure(other)).toBe(other);
  });

  it('does not throw on a non-Error rejection', () => {
    // MediaPipe rejects with strings in places, and a crash inside the error path
    // would replace a diagnosable failure with an undiagnosable one.
    expect(explainModalityFailure('plain string')).toBe('plain string');
    expect(explainModalityFailure(undefined)).toBe(undefined);
  });
});
