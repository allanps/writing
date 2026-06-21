# A small harness for calibrating an LLM-as-judge before it gates anything

## The problem

An LLM-as-judge is convenient. Hand it a model output and some criteria, get back pass or fail. The catch is that a judge is itself a model, with its own failure modes. Most setups never answer the one question that makes the judge worth running: do you agree with it?

`llm-evals-mini` is a small harness that tries to answer that question before the judge gates anything. It does three things. It calibrates a judge against human labels with Cohen's kappa. It validates structured outputs against a schema. And it exposes a CI gate that fails when the score drops below a threshold.

## The approach

Calibration runs the judge over a fixture of `(prompt, output, human_label)` rows. The judge never sees `human_label`, so the comparison is honest. It then scores agreement two ways: plain accuracy, and Cohen's kappa.

Kappa matters because accuracy can mislead when one label dominates. A judge that always says "pass" scores high accuracy on a dataset that is mostly passing, while telling you nothing. Kappa corrects for the agreement you would expect by chance. The implementation computes the expected agreement under independence directly:

```python
expected = 0.0
for label in _LABELS:
    p_human = human.count(label) / n
    p_judge = judge.count(label) / n
    expected += p_human * p_judge

if expected >= 1.0:
    return 1.0 if observed == 1.0 else 0.0
return (observed - expected) / (1.0 - expected)
```

The degenerate case is handled explicitly. When both raters use a single label for every item, `expected` hits 1.0 and the formula would divide by zero, so it returns 1.0 only if they actually agreed everywhere, else 0.0.

## How it is built

The judge goes through a one-method interface, `judge(example) -> Judgment`. `FakeJudge` implements it with keyword rules so the bundled metric reproduces offline with no API key. `AnthropicJudge` implements the same interface against Claude (`claude-sonnet-4-6` by default), with the SDK imported lazily so the package imports fine without it. Swapping judges is a one-line change at the call site.

The real judge asks for JSON constrained to a schema:

```python
_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string", "enum": ["pass", "fail"]},
        "reason": {"type": "string"},
    },
    "required": ["label", "reason"],
    "additionalProperties": False,
}
```

`Judgment.__post_init__` rejects any label that is not `pass` or `fail`, so a malformed decision fails loudly rather than slipping through.

Structured-output validation is separate. `validate_outputs(outputs, schema)` runs a batch of JSON strings or dicts through a Pydantic model and reports a violation rate. A string that is not valid JSON counts as a violation. A dict that fails schema validation counts as a violation. The report carries the per-output error list and the rate.

The CI gate is the `evals gate --min-score X` command. It runs calibration and exits 1 when the chosen score (kappa by default, accuracy optional) is below the threshold. The bundled GitHub workflow wires it in after lint and tests:

```yaml
- name: Regression gate (offline, no API key)
  run: evals gate --min-score 0.8
```

Every `calibrate` run also writes `evals_report.md` with the calibration table, a confusion matrix, and per-example judgments. The report labels kappa with the Landis & Koch (1977) bands: `slight`, `fair`, `moderate`, `substantial`, `almost perfect`.

## What was measured

The headline numbers come from running the bundled 20-row fixture (`fixtures/support_replies.jsonl`, with human pass/fail labels) through the deterministic `FakeJudge`:

- examples: 20
- accuracy: 1.000
- Cohen's kappa: 1.000

The suite has 28 tests. They pass offline with no network and no key. The fixture is deliberately small and clean so the fake judge reproduces the labels exactly, which keeps the headline stable across machines.

## What calibration buys you, and where it breaks

Calibration gives you a number you can point to. Instead of "the judge looks reasonable," you can say "the judge agrees with our labels at kappa 0.7 on this set," and you can gate CI on that not regressing.

The honest limitations, which the README states plainly:

- A kappa of 1.0 here means the keyword `FakeJudge` matches a small clean set, not that judging is solved. Point `calibrate --judge anthropic` at your own labeled data and kappa drops below 1.0, which is where the number starts telling you something.
- Calibration is only as honest as the human labels. Bad labels give a confident, wrong kappa.
- It models one judge against one rater of ground truth. There is no inter-annotator agreement among multiple humans.
- Only two labels (`pass`/`fail`). No multi-class or graded scoring.
- Schema validation checks structure, not meaning. A well-formed but wrong answer passes.

What I would change next: the two-label restriction is the first ceiling you hit on real data, where "pass with caveats" is common, so graded labels and a weighted kappa would be the next step. And calibration on a single ground-truth rater is a starting point, not a finish line. The moment you have two annotators, you want their agreement in the same report so you know whether the judge or the labels are the weaker link.
