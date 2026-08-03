// Minimal protobuf reader for ONNX headers.
//
// Exports come in two layouts. Most keep weights in a sidecar .onnx_data, which
// leaves the .onnx small (5,621 bytes for gemma-4's embed_tokens) and lists every
// tensor's size as external_data — so a Range request over the header reveals the
// largest tensor without downloading any weights. Others embed weights directly in
// the .onnx (Qwen3-1.7B ships 1360MiB), where no external_data exists and this
// parser returns nothing; inspect.ts falls back to the file length there.
//
// Field numbers come from onnx.proto:
//   ModelProto.graph = 7, GraphProto.initializer = 5,
//   TensorProto.name = 8, TensorProto.external_data = 13,
//   StringStringEntryProto.key = 1, .value = 2

const FIELD = {
  GRAPH: 7,
  INITIALIZER: 5,
  NAME: 8,
  EXTERNAL_DATA: 13,
  KEY: 1,
  VALUE: 2,
} as const;

const WIRE = { VARINT: 0, I64: 1, LEN: 2, I32: 5 } as const;

export interface ExternalTensor {
  name: string;
  bytes: number;
  location: string;
}

export interface OnnxHeader {
  tensors: ExternalTensor[];
  largestTensorBytes: number;
  totalExternalBytes: number;
}

function readVarint(bytes: Uint8Array, pos: number): [number, number] {
  let result = 0;
  let shift = 0;
  let i = pos;
  while (i < bytes.length) {
    const byte = bytes[i++]!;
    // Multiplication rather than << to stay exact above 2^31.
    result += (byte & 0x7f) * Math.pow(2, shift);
    if ((byte & 0x80) === 0) return [result, i];
    shift += 7;
    if (shift > 63) throw new Error('varint too long');
  }
  throw new Error('unexpected end of buffer');
}

type Visitor = (field: number, start: number, end: number) => void;

// Walks length-delimited fields, returning quietly on malformed input so a
// truncated Range response degrades to fewer results rather than an exception.
function eachField(bytes: Uint8Array, start: number, end: number, visit: Visitor): void {
  let i = start;
  // The whole body is guarded, not just the tag: a Range response can end
  // mid-varint, and readVarint throws there. Returning leaves whatever was
  // already visited intact, which is the degrade-to-fewer-tensors behaviour
  // callers rely on.
  try {
    while (i < end) {
      let tag: number;
      let next: number;
      [tag, next] = readVarint(bytes, i);
      if (next > end) return;
      const field = tag >>> 3;
      const wire = tag & 7;
      i = next;
      if (wire === WIRE.LEN) {
        let len: number;
        [len, i] = readVarint(bytes, i);
        if (i + len > end) return;
        visit(field, i, i + len);
        i += len;
      } else if (wire === WIRE.VARINT) {
        [, i] = readVarint(bytes, i);
      } else if (wire === WIRE.I32) {
        i += 4;
      } else if (wire === WIRE.I64) {
        i += 8;
      } else {
        return;
      }
    }
  } catch {
    return;
  }
}

export function parseOnnxHeader(bytes: Uint8Array): OnnxHeader {
  const decoder = new TextDecoder();
  const tensors: ExternalTensor[] = [];

  eachField(bytes, 0, bytes.length, (field, start, end) => {
    if (field !== FIELD.GRAPH) return;
    eachField(bytes, start, end, (graphField, gStart, gEnd) => {
      if (graphField !== FIELD.INITIALIZER) return;

      let name = '';
      const entries: Record<string, string> = {};

      eachField(bytes, gStart, gEnd, (tField, tStart, tEnd) => {
        if (tField === FIELD.NAME) {
          name = decoder.decode(bytes.subarray(tStart, tEnd));
        } else if (tField === FIELD.EXTERNAL_DATA) {
          let key = '';
          let value = '';
          eachField(bytes, tStart, tEnd, (eField, eStart, eEnd) => {
            if (eField === FIELD.KEY) key = decoder.decode(bytes.subarray(eStart, eEnd));
            else if (eField === FIELD.VALUE) value = decoder.decode(bytes.subarray(eStart, eEnd));
          });
          if (key) entries[key] = value;
        }
      });

      const length = entries['length'];
      if (length !== undefined) {
        tensors.push({
          name,
          bytes: Number(length),
          location: entries['location'] ?? '',
        });
      }
    });
  });

  return {
    tensors,
    largestTensorBytes: tensors.reduce((max, t) => (t.bytes > max ? t.bytes : max), 0),
    totalExternalBytes: tensors.reduce((sum, t) => sum + t.bytes, 0),
  };
}
