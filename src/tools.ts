// Tool calls arrive as text in the middle of a generation, so parsing is a
// separate pure step rather than a change to the streaming contract.

export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
}

export interface ToolCallFormat {
  /** Wraps each call, e.g. ['<tool_call>', '</tool_call>']. */
  delimiters: [string, string];
  /**
   * How the body between the delimiters is written.
   *
   * - 'json': `{"name": …, "arguments": {…}}` — Qwen3, and the shape Qwen2.5 and
   *   Granite 4.0 also emit, though neither has a catalog entry here.
   * - 'xml': `<function=name><parameter=key>value</parameter></function>` —
   *   Qwen3.5. Values come back as strings, since the markup carries no types.
   * - 'gemma4': `call:name{key:<|"|>str<|"|>,n:3}` — gemma-4. Types ARE carried,
   *   since only strings are wrapped.
   *
   * **'gemma4' needs different input.** Its delimiters and its `<|"|>` string
   * marker are *special* tokens, so `skip_special_tokens: true` erases them before
   * this function could see them. Pass text decoded with `skip_special_tokens:
   * false` — `GenerateOptions.onRawText` supplies exactly that. The other two
   * encodings work on either.
   */
  encoding: 'json' | 'xml' | 'gemma4';
}

// Enough of the offending text to identify it, not enough to flood the console.
const excerpt = (s: string) => (s.length > 80 ? `${s.slice(0, 80)}…` : s);

function warn(reason: string): void {
  console.warn(`[in-browser-llm] ${reason}`);
}

// Returns null on anything unparseable; the caller warns and drops the call.
function parseJsonBody(body: string): ToolCall | null {
  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    warn(`discarded a tool call: body is not valid JSON: ${excerpt(body)}`);
    return null;
  }
  if (typeof parsed !== 'object' || parsed === null) {
    warn(`discarded a tool call: body is not an object: ${excerpt(body)}`);
    return null;
  }
  const { name, arguments: args } = parsed as Record<string, unknown>;
  if (typeof name !== 'string') {
    warn(`discarded a tool call: "name" must be a string: ${excerpt(body)}`);
    return null;
  }
  // A model asked for a no-argument tool often omits the key entirely.
  const resolved = args ?? {};
  if (typeof resolved !== 'object' || resolved === null || Array.isArray(resolved)) {
    warn(`discarded tool call "${name}": "arguments" must be an object`);
    return null;
  }
  return { name, arguments: resolved as Record<string, unknown> };
}

// Values stay strings: the XML markup carries no types, so inferring them would be a
// guess. A caller with a schema can coerce; this function cannot know. Contrast the
// gemma4 encoding below, which does recover types — because its dialect marks strings
// explicitly rather than leaving the question open.
//
// Nor is the string necessarily a form JSON accepts: asked for a `boolean`, Qwen3.5 wrote
// `True`, so `value === 'true'` is false for a value it meant as true. Coercion belongs to
// the caller who knows the declared type; folding case here would still be guessing.
const FUNCTION_RE = /<function=([^>]+)>/;
const PARAMETER_RE = /<parameter=([^>]+)>\n?([\s\S]*?)\n?<\/parameter>/g;

function parseXmlBody(body: string): ToolCall | null {
  const fn = FUNCTION_RE.exec(body);
  if (!fn?.[1]) {
    warn(`discarded a tool call: no <function=…> in body: ${excerpt(body)}`);
    return null;
  }
  const args: Record<string, unknown> = {};
  for (const m of body.matchAll(PARAMETER_RE)) {
    if (m[1]) args[m[1]] = m[2] ?? '';
  }
  return { name: fn[1], arguments: args };
}

// gemma-4's string marker, `escape_token`, id 52. It opens AND closes a string, so a
// string's contents are unambiguous even when they contain , { } or [ ] — which is
// why this is a scanner rather than a regex or a transform into JSON.
const GEMMA4_QUOTE = '<|"|>';

// From the export's own tokenizer_config.json `response_schema`:
//   x-regex "call\:(?P<name>\w+)(?P<arguments>\{.*\})"
const GEMMA4_CALL_RE = /^call:(\w+)\s*(\{[\s\S]*\})$/;

interface Scanner {
  body: string;
  at: number;
}

// Returns undefined on malformed input, which propagates up to a discarded call.
// undefined rather than null because null is a legitimate parsed value here: gemma-4
// renders a null argument as an empty value, `{a:}`.
function scanValue(s: Scanner): unknown {
  if (s.body.startsWith(GEMMA4_QUOTE, s.at)) {
    const from = s.at + GEMMA4_QUOTE.length;
    const to = s.body.indexOf(GEMMA4_QUOTE, from);
    if (to === -1) return undefined;
    s.at = to + GEMMA4_QUOTE.length;
    return s.body.slice(from, to);
  }

  const open = s.body[s.at];
  if (open === '{' || open === '[') {
    const isObject = open === '{';
    const close = isObject ? '}' : ']';
    s.at += 1;
    const obj: Record<string, unknown> = {};
    const arr: unknown[] = [];

    // Handles the empty forms `{}` and `[]` before looking for any member.
    while (s.body[s.at] !== close) {
      if (s.at >= s.body.length) return undefined;

      if (isObject) {
        // Keys are bare identifiers, never quoted — `city:` not `<|"|>city<|"|>:`.
        const colon = s.body.indexOf(':', s.at);
        if (colon === -1) return undefined;
        const key = s.body.slice(s.at, colon).trim();
        if (!key) return undefined;
        s.at = colon + 1;
        const value = scanValue(s);
        if (value === undefined) return undefined;
        obj[key] = value;
      } else {
        const value = scanValue(s);
        if (value === undefined) return undefined;
        arr.push(value);
      }

      if (s.body[s.at] === ',') s.at += 1;
      else if (s.body[s.at] !== close) return undefined;
    }
    s.at += 1;
    return isObject ? obj : arr;
  }

  // A bare token: a number, a boolean, or nothing at all. Ends at whatever
  // terminates the enclosing container.
  let end = s.at;
  while (end < s.body.length && !',}]'.includes(s.body[end]!)) end += 1;
  const raw = s.body.slice(s.at, end).trim();
  s.at = end;

  // `{a:}` is how a null argument renders — an empty value, not a missing key.
  if (raw === '') return null;
  if (raw === 'true') return true;
  if (raw === 'false') return false;
  if (raw === 'null') return null;
  const num = Number(raw);
  // Unquoted and not a literal: the dialect has no such form, so treating it as a
  // string would invent a type the model did not ask for.
  return Number.isNaN(num) ? undefined : num;
}

/**
 * Parses gemma-4's tool-call dialect: `call:name{city:<|"|>Seoul<|"|>,nights:3}`.
 *
 * Unlike the XML encoding, types ARE recoverable: the dialect wraps strings in
 * `<|"|>` and leaves numbers, booleans and containers bare. Verified by rendering
 * the real template — `{a:'x'}` becomes `a:<|"|>x<|"|>` while `{a:3}` becomes `a:3`.
 *
 * Requires text decoded with `skip_special_tokens: false`; `<|"|>` is a special
 * token (id 52) and vanishes otherwise, at which point `{a:3}` and `{a:"3"}` are
 * indistinguishable and this parser would silently return the wrong types.
 */
function parseGemma4Body(body: string): ToolCall | null {
  const match = GEMMA4_CALL_RE.exec(body.trim());
  if (!match?.[1] || !match[2]) {
    warn(`discarded a tool call: not a gemma-4 call:name{…} body: ${excerpt(body)}`);
    return null;
  }

  const scanner: Scanner = { body: match[2], at: 0 };
  const args = scanValue(scanner);
  if (args === undefined || typeof args !== 'object' || args === null || Array.isArray(args)) {
    warn(`discarded tool call "${match[1]}": arguments are not an object: ${excerpt(body)}`);
    return null;
  }
  // Trailing content means the scan stopped early, so the arguments are incomplete
  // rather than merely odd — a partially parsed call is worse than none.
  if (scanner.at !== scanner.body.length) {
    warn(`discarded tool call "${match[1]}": trailing content after arguments`);
    return null;
  }

  return { name: match[1], arguments: args as Record<string, unknown> };
}

/**
 * Extracts tool calls from generated text. Never throws.
 *
 * A small model produces malformed JSON and invents function names, so every
 * rejection drops that one call and warns. An empty result means either "no tool
 * wanted" or "everything was rejected" — the warnings distinguish them, and the
 * caller treats both as an ordinary answer.
 *
 * Pass `allowed` to reject names outside the tool list supplied to the model.
 */
export function parseToolCalls(
  text: string,
  format: ToolCallFormat,
  allowed?: string[],
): ToolCall[] {
  const [open, close] = format.delimiters;
  const calls: ToolCall[] = [];

  let cursor = 0;
  for (;;) {
    const start = text.indexOf(open, cursor);
    if (start === -1) break;
    const bodyStart = start + open.length;
    const end = text.indexOf(close, bodyStart);
    // An unterminated call means generation stopped mid-emission, usually at the
    // token cap. There is nothing to salvage.
    if (end === -1) break;
    cursor = end + close.length;

    const body = text.slice(bodyStart, end).trim();
    const parse =
      format.encoding === 'xml'
        ? parseXmlBody
        : format.encoding === 'gemma4'
          ? parseGemma4Body
          : parseJsonBody;
    const call = parse(body);
    if (!call) continue;

    if (allowed && !allowed.includes(call.name)) {
      warn(`discarded tool call "${call.name}": not in the supplied tool list`);
      continue;
    }

    calls.push(call);
  }

  return calls;
}
