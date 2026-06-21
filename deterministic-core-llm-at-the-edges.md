# A deterministic core with the LLM at the edges

## The problem

A common way to build an agent is to put the rules in the prompt. Something like: classify the ticket, send security issues to the security team, redact PII, never auto-close an account issue without a human. Every one of those is a request the model may or may not honor. A jailbreak, a bad sample, schema drift, or a model upgrade can break any of them, and you often find out in production.

The model is good at reading a ticket and proposing a category. It is not a good place to keep a guarantee.

## The approach

Keep the control flow and the rules in ordinary code. Call the LLM only at named, typed edges. Treat whatever the model returns as a proposal, and validate it before it touches anything that matters.

`agentcore` is a small template that does this. The demo domain is support-ticket triage. The agent is a linear state machine: `NEW -> CATEGORIZED -> ROUTED -> DRAFTED -> CLOSED`. The model is asked three narrow questions (what category, what priority, draft a reply) and nothing it answers is trusted until code checks it.

## How it is built

Three files carry the weight.

`core.py` owns the guarantees: the transition table, the routing map, the SLA table, the human-review rule, and PII redaction. Its functions are pure, `(Decision, ...) -> Decision`, so each rule is unit-testable without the model. State changes go through one chokepoint, which raises on an illegal jump:

```python
def transition(decision: Decision, dst: State) -> Decision:
    if not can_transition(decision.state, dst):
        raise TransitionError(
            f"illegal transition {decision.state.value} -> {dst.value}"
        )
    return Decision(state=dst, ...)
```

Model output is coerced to a fixed enum. An unknown string can never reach routing:

```python
def coerce_category(raw: object) -> Category:
    if isinstance(raw, Category):
        return raw
    if isinstance(raw, str):
        key = raw.strip().lower().replace("-", "_").replace(" ", "_")
        for member in Category:
            if key == member.value or key == member.name.lower():
                return member
    return Category.OTHER
```

Some rules escalate but never weaken. A security ticket is forced to at least HIGH, sent to the security queue, and flagged for human review, regardless of what the model said:

```python
if category is Category.SECURITY:
    priority = _max_priority(priority, Priority.HIGH)
    queue = _QUEUE_BY_CATEGORY[Category.SECURITY]
```

`edges.py` holds the three LLM seams. Each builds a prompt and a small JSON schema, calls the client, and returns the raw value. The edges decide nothing.

`agent.py` is the driver. Read `triage` top to bottom and every edge output is immediately handed to a core function that validates it. There is no path from edge to decision that skips the core.

The LLM seam is a one-method `Protocol`. A `FakeLLMClient` returns canned responses for offline tests. `AnthropicLLMClient` is the real default (Claude Sonnet by default, with the `anthropic` import done lazily so the package imports with no SDK and no key).

## What was measured

The reliability claim is a test, not a sentence in a README. The headline test runs the full triage pipeline against a Cartesian product of bad model outputs: 8 category values, 6 priority values, and 7 reply values, which is 336 runs. The bad values cover unknown strings, wrong types, empty values, `None`, and injection-style junk like `"drop table tickets;"`.

After each run it checks six invariants: the ticket ends in `CLOSED`, category and priority are valid enum members, the queue is known, the SLA is a positive integer, security tickets are escalated, and the reply is a redacted, length-capped string.

| Metric | Value |
| --- | --- |
| Adversarial model outputs that violated a core rule | 0 / 336 |
| Total tests passing, no network, no API key | 449 |

```
449 passed in 3.26s
```

Both numbers reproduce from a clean `pip install -e ".[dev]" && pytest`.

## Limitations

This pattern moves correctness into code, which means you write the rules. If a rule is wrong, the core enforces it faithfully and wrongly. The point is that the rule is explicit, versioned, and tested, not that it is automatically correct.

It does not make the model's judgment better. If the model picks the wrong category for an ambiguous ticket, the core will route a valid-but-wrong category cleanly. The guarantees are structural (legal state, known queue, positive SLA, no leaked PII, security escalated), not semantic.

The PII redaction is regex for emails, card-like digit runs, and US SSNs. It is a guardrail, not a DLP layer, and it will miss formats it does not match.

`FakeLLMClient` is what makes the 336-run test deterministic. It does not model latency, token limits, or streaming, so the test says nothing about those.

This is a linear pipeline. Branching workflows, retries, and tool loops are out of scope for the template. The core/edge split should extend to them, but I have not implemented that here.

What I would change next: make the rule tables data rather than code so they can be reviewed without a diff to logic, and add a property-based generator for the adversarial inputs instead of a hand-listed product, so the 336 grows on its own.