# docquery: a small RAG service that cites its sources and refuses when retrieval is weak

## The problem

A RAG system that always answers is a system that sometimes makes things up. When retrieval returns nothing useful, the honest output is "I don't know," not a fluent guess. And when it does answer, you want to know which retrieved chunk each claim came from.

`docquery` is a small RAG service built around those two requirements: inline `[1][2]` citations on every answer, and a refusal path that triggers when retrieval is weak. It is eval-first, so the claims here come from code in the repo, not from a demo I ran once.

## The approach

The pipeline is five steps, one module each: ingest and chunk, embed, index, synthesize, then ground and refuse. The last two are where the two requirements live.

**Retrieval.** Chunks are embedded with sentence-transformers `all-MiniLM-L6-v2`, stored unit-normalized in a numpy matrix. A query is one matrix-vector product, and top-k is exact:

```python
q = self.embedder.encode([question])[0]
scores = self._matrix @ q          # cosine, vectors are unit-normalized
top_idx = np.argpartition(-scores, k - 1)[:k]
top_idx = top_idx[np.argsort(-scores[top_idx])]
```

No FAISS, no external service. The dependency for retrieval is numpy.

**The refusal threshold.** Before any LLM call, the pipeline checks the top-1 similarity. If it is below the threshold (default `0.25`), it refuses without asking the model anything:

```python
top_score = retrieved[0].score if retrieved else 0.0
if not retrieved or top_score < self.threshold:
    return Answer(text=REFUSAL, retrieved=retrieved, refused=True,
                  top_score=top_score, cited_indices=[])
```

This is a cheap, deterministic gate that runs before synthesis, so an out-of-scope question never reaches the model. The model also gets a system prompt instructing it to reply with the exact refusal string if the context does not contain the answer. So there are two layers: the score gate and the prompt.

**Citation mapping.** The context block is numbered before it goes to the model, and the answer is expected to cite those numbers. After synthesis, citations are parsed back out and validated against the retrieved set:

```python
for match in re.findall(r"\[(\d+)\]", text):
    idx = int(match)
    if 1 <= idx <= n and idx not in found:
        found.append(idx)
```

Only distinct, in-range indices survive. A `[9]` when four chunks were retrieved is dropped, so a citation index always resolves to a real source.

The LLM is behind a one-method interface. `AnthropicClient` (default `claude-sonnet-4-6`) is the real adapter, with the SDK imported lazily so the package imports with no SDK and no key. `FakeClient` is deterministic and extractive: it stitches the first sentence of the top chunks together and appends `[1]`, `[2]`. That fake powers the tests and the offline eval, which is what makes the numbers reproducible.

## What was measured

The eval runs the full pipeline against `fixtures/qa.json`: 8 in-scope questions, each labeled with its supporting document, plus 4 out-of-scope questions. It runs offline with the Fake client and local embeddings, and writes the table to `eval_report.md`. On the MiniLM backend:

| Metric | Score |
| --- | --- |
| Retrieval recall@k (k=4) | 1.000 |
| Refusal accuracy | 1.000 |
| Citation accuracy | 0.875 |
| Answer keyword accuracy | 0.625 |

Recall@k asks whether the labeled source document appeared in the top-4. Refusal accuracy is the share of out-of-scope questions refused. Citation accuracy checks that every cited chunk was drawn from the question's supporting document, counting a question as a miss if any cited index points elsewhere:

```python
for idx in cited_indices:
    chunk = retrieved[idx - 1].chunk
    if chunk.source != relevant_source:
        return False
```

So 0.875 means one of the eight in-scope answers cited a chunk from the wrong document. (It is the "what percentage of Earth's surface is ocean" question, which pulls in a chunk from the photosynthesis doc.) The repo ships 35 tests covering chunking, retrieval ordering, citation parsing, the refusal threshold, and the eval producing these numbers. They run with no network and no key, forcing the offline hashing embedder.

## Honest limitations

A few things are worth stating plainly, and they are in the README too:

- **Answer keyword accuracy is 0.625 offline.** The `FakeClient` is extractive: it takes the first sentence of each top chunk. When the answer phrase lives in a later sentence, the keyword is absent even though retrieval and citations are fine. A real model phrases the result directly and would raise this number, but the repo reports the offline figure because it is the one anyone can reproduce without a key.
- **The offline fallback embedder is lexical.** If MiniLM cannot be downloaded, retrieval falls back to a deterministic hashing embedder that matches on token overlap. It keeps the pipeline running with zero network, but paraphrases with no shared words retrieve poorly. It exists for reproducibility, not for quality.
- **Retrieval is brute-force cosine over an in-memory matrix.** Fine for small, self-contained corpora; not built for millions of chunks.
- **The refusal threshold is a single global cutoff** tuned on this fixture. A different corpus or embedder may want a different value. It is a constructor argument and a CLI flag, so it is at least easy to retune, but it is not adaptive.

If I extended this, the first change would be a per-query confidence signal instead of one global threshold, and a labeled set large enough to tune that cutoff on a held-out split rather than on the same fixture I report on. The structure already supports it: the threshold is a parameter, and the eval is the place to measure whether a new value helps.

MIT licensed, provider-agnostic, with a deterministic fake for offline tests.