"""Agentic Kimi K2.6 replay benchmark modified for SGLang."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import operator
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

# Modified to use the unpacked file
DATA_PATH = Path(__file__).with_name('data.jsonl')
# Modified to match the model name served by SGLang
MODEL = '/models'
API_KEY = 'dummy'
PARALLELISM = 64
TIMEOUT_S = 1800.0
REQUEST_BODY = {'temperature': 0.6, 'top_p': 0.95, 'max_tokens': 4096}
PERCENTILES = (50, 90, 95, 99)

type Item = dict[str, Any]
type Trace = tuple[str, int, list[Item]]


def _url(endpoint: str) -> str:
  endpoint = endpoint.rstrip('/')
  if endpoint.endswith('/chat/completions'):
    return endpoint
  suffix = '' if endpoint.endswith('/v1') else '/v1'
  return f'{endpoint}{suffix}/chat/completions'


def _route(sequence_id: str, endpoint_count: int) -> int:
  digest = hashlib.sha256(sequence_id.encode()).digest()
  return int.from_bytes(digest[:8], 'big') % endpoint_count


def _load(endpoint_count: int) -> list[Trace]:
  grouped: dict[str, list[Item]] = {}
  # Modified to use standard open for unpacked jsonl
  with open(DATA_PATH, 'rt', encoding='utf-8') as f:
    for line in f:
      if line.strip():
        item = json.loads(line)
        grouped.setdefault(item['sequence_id'], []).append(item)
  return [
    (
      sequence_id,
      _route(sequence_id, endpoint_count),
      sorted(steps, key=operator.itemgetter('step_index')),
    )
    for sequence_id, steps in grouped.items()
  ]


def _num(mapping: Item, key: str) -> int:
  return int(mapping.get(key) or 0)


def _stats(values: Sequence[float | int]) -> Item:
  if not values:
    return {'mean': None, **{f'p{p}': None for p in PERCENTILES}}
  ordered = sorted(float(value) for value in values)
  return {
    'mean': sum(ordered) / len(ordered),
    **{
      f'p{p}': ordered[int((len(ordered) - 1) * p / 100)] for p in PERCENTILES
    },
  }


def _tokens(results: Sequence[Item]) -> tuple[int, int, int, int]:
  prompt = sum(item['prompt_tokens'] for item in results)
  completion = sum(item['completion_tokens'] for item in results)
  cached = sum(item['cached_prompt_tokens'] for item in results)
  return prompt, completion, cached, prompt + completion


async def _run_trace(
  client: httpx.AsyncClient,
  urls: Sequence[str],
  trace: Trace,
  results: list[Item],
  failures: list[Item],
) -> None:
  sequence_id, endpoint_index, steps = trace
  url = urls[endpoint_index]
  for step in steps:
    body = {'model': MODEL, 'messages': step['messages'], **REQUEST_BODY}
    if step.get('tools'):
      body['tools'] = step['tools']
      
    prompt_chars = sum(len(m.get('content', '')) for m in step['messages'])
    tools_count = len(step.get('tools', []))
    
    started = time.perf_counter()
    try:
      response = await client.post(url, json=body)
      elapsed_s = time.perf_counter() - started
      response.raise_for_status()
      usage = response.json().get('usage') or {}
      details = usage.get('prompt_tokens_details') or {}
      results.append({
        'endpoint_index': endpoint_index,
        'latency_s': elapsed_s,
        'prompt_tokens': _num(usage, 'prompt_tokens'),
        'completion_tokens': _num(usage, 'completion_tokens'),
        'cached_prompt_tokens': _num(details, 'cached_tokens'),
      })
      print(f"Success: Trace {sequence_id}, Step {step.get('step_index')}, Prompt Chars: {prompt_chars}, Tools: {tools_count}, Latency: {elapsed_s:.2f}s")
    except Exception as e:  # noqa: BLE001
      failures.append({
        'sequence_id': sequence_id,
        'step_index': step.get('step_index'),
        'error': f'{type(e).__name__}: {e}',
      })
      print(f"Failure: Trace {sequence_id}, Step {step.get('step_index')}, Prompt Chars: {prompt_chars}, Tools: {tools_count}, Error: {type(e).__name__}: {e}")


def _endpoint_summaries(
  endpoints: Sequence[str], results: Sequence[Item]
) -> list[Item]:
  summaries = []
  for index, endpoint in enumerate(endpoints):
    matches = [item for item in results if item['endpoint_index'] == index]
    prompt, completion, cached, _ = _tokens(matches)
    summaries.append({
      'endpoint': endpoint,
      'completed': len(matches),
      'prompt_tokens': prompt,
      'completion_tokens': completion,
      'cached_prompt_tokens': cached,
      'prompt_cache_hit_rate': cached / prompt if prompt else None,
    })
  return summaries


def _summary(
  endpoints: Sequence[str],
  traces: Sequence[Trace],
  results: Sequence[Item],
  failures: Sequence[Item],
  started_at: dt.datetime,
  wall_s: float,
) -> Item:
  prompt, completion, cached, total = _tokens(results)
  wall_s = max(wall_s, 1e-9)
  return {
    'started_at': started_at.isoformat(),
    'completed_at': dt.datetime.now(dt.UTC).isoformat(),
    'data_path': str(DATA_PATH),
    'model': MODEL,
    'endpoints': list(endpoints),
    'request_body': REQUEST_BODY,
    'parallelism': PARALLELISM,
    'request_timeout_s': TIMEOUT_S,
    'num_sequences': len(traces),
    'num_requests_total': sum(len(steps) for _, _, steps in traces),
    'num_requests_completed': len(results),
    'num_requests_failed': len(failures),
    'wall_clock_seconds': wall_s,
    'latency_seconds': _stats([item['latency_s'] for item in results]),
    'prompt_tokens': _stats([item['prompt_tokens'] for item in results]),
    'completion_tokens': _stats([
      item['completion_tokens'] for item in results
    ]),
    'tokens': {
      'prompt': prompt,
      'completion': completion,
      'total': total,
      'cached_prompt': cached,
      'prompt_cache_hit_rate': cached / prompt if prompt else None,
    },
    'throughput': {
      'requests_per_second': len(results) / wall_s,
      'prompt_tokens_per_second': prompt / wall_s,
      'completion_tokens_per_second': completion / wall_s,
      'total_tokens_per_second': total / wall_s,
    },
    'endpoint_summaries': _endpoint_summaries(endpoints, results),
    'sample_failures': list(failures[:25]),
  }


async def _run(endpoints: Sequence[str]) -> Item:
  urls = [_url(endpoint) for endpoint in endpoints]
  traces = _load(len(urls))
  if not traces:
    raise ValueError(f'no benchmark rows loaded from {DATA_PATH}')

  started_at = dt.datetime.now(dt.UTC)
  started = time.perf_counter()
  results: list[Item] = []
  failures: list[Item] = []
  limits = httpx.Limits(
    max_connections=PARALLELISM, max_keepalive_connections=PARALLELISM
  )
  async with httpx.AsyncClient(
    headers={'Authorization': f'Bearer {API_KEY}'},
    limits=limits,
    timeout=TIMEOUT_S,
  ) as client:
    trace_iterator = iter(traces)

    async def worker() -> None:
      for trace in trace_iterator:
        await _run_trace(client, urls, trace, results, failures)

    await asyncio.gather(
      *(worker() for _ in range(min(PARALLELISM, len(traces))))
    )

  return _summary(
    endpoints,
    traces,
    results,
    failures,
    started_at,
    time.perf_counter() - started,
  )


def main(argv: Sequence[str] | None = None) -> None:
  """Parse endpoint URLs and run the benchmark."""
  global PARALLELISM, TIMEOUT_S
  parser = argparse.ArgumentParser(description='Replay the bundled benchmark.')
  parser.add_argument('endpoints', nargs='+', help='Endpoint URL(s).')
  parser.add_argument('--parallelism', type=int, default=64, help='Number of concurrent requests.')
  parser.add_argument('--timeout', type=float, default=1800.0, help='Request timeout in seconds.')
  args = parser.parse_args(argv)
  PARALLELISM = args.parallelism
  TIMEOUT_S = args.timeout
  payload = asyncio.run(_run(args.endpoints))
  output = Path(f'results-{dt.datetime.now(dt.UTC):%Y%m%d-%H%M%S}.json')
  with output.open('w', encoding='utf-8') as f:
    json.dump(payload, f, indent=2)
    f.write('\n')
  print(output)
  print("=== BENCHMARK RESULTS ===")
  print(json.dumps(payload, indent=2))


if __name__ == '__main__':
  main()
