import datasets as hf_datasets
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from transformers import pipeline

FUZZ_METRICS = [
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "WRatio",
    "QRatio",
]


def compute_without_references(metric_name, predictions):
    def _get_hf_model(metric_name):
        try:
            return pipeline(metric_name)
        except KeyError:
            return pipeline(model=metric_name)

    model = _get_hf_model(metric_name)
    return model(predictions)


def compute_with_references(metric_name, references, predictions):
    if metric_name in FUZZ_METRICS:
        return _compute_with_references_fuzz(
            metric_name, references, predictions
        )
    elif metric_name == "levenshtein":
        return np.mean(
            [
                Levenshtein.distance(reference, prediction)
                for reference, prediction in zip(references, predictions)
            ]
        )
    else:
        # Try huggingface :)
        return _compute_with_references_hf(
            metric_name, references, predictions
        )


def _compute_with_references_fuzz(metric_name, references, predictions):
    fn = getattr(fuzz, metric_name)
    return np.mean(
        [
            fn(reference, prediction)
            for reference, prediction in zip(references, predictions)
        ]
    )


def _compute_with_references_hf(metric_name, references, predictions):
    def _get_metric_name_for_hf_load(metric_name):
        if "rouge" in metric_name:
            return metric_name.split("-")[0]
        else:
            return metric_name

    def _compute(metric_name):
        metric_name_for_hf_load = _get_metric_name_for_hf_load(metric_name)
        return hf_datasets.load_metric(metric_name_for_hf_load).compute(
            predictions=predictions, references=references
        )

    def _extract(metric_name, results):
        if metric_name.startswith("rouge"):
            key = "".join(metric_name.split("-")[:-1])
            if key not in results:
                return results["rouge1"].mid.fmeasure
            elif "precision" in metric_name:
                return results[key].mid.precision
            elif "recall" in metric_name:
                return results[key].mid.recall
            elif "fmeasure" in metric_name:
                return results[key].mid.fmeasure
            else:
                raise ValueError(f"Unknown metric_name: {metric_name}")
        else:
            # General case
            key = {
                "sacrebleu": "score",
                "ter": "score",
            }.get(metric_name, metric_name)

            try:
                return results[key]
            except KeyError:
                print(f"[KEYERROR] {results=}")
                for k, v in results.items():
                    print(f"{k}: {v}")

    results = _compute(metric_name)
    value = _extract(metric_name, results)
    return value


def run_tests():
    predictions = ["hello there general kenobi", "foo bar foobar"]
    predictions_tokenized = [p.split() for p in predictions]
    references = [["hello there general kenobi"], ["foo bar foobar"]]
    references_tokenized = [[r.split() for r in rs] for rs in references]
    assert (
        compute_with_references(
            "bleu", references_tokenized, predictions_tokenized
        )
        == 1.0
    )

    predictions = ["hello there general kenobi", "foo bar foobar"]
    references = [["hello there general kenobi"], ["foo bar foobar"]]
    assert np.isclose(
        compute_with_references("sacrebleu", references, predictions), 100
    )

    predictions = ["hello there general kenobi", "foo bar foobar"]
    predictions_tokenized = [p.split() for p in predictions]
    references = [["hello there general kenobi"], ["foo bar foobar"]]
    references_tokenized = [[r.split() for r in rs] for rs in references]

    assert compute_with_references("ter", references, predictions) == 0

    predictions = ["hello there general kenobi", "foo bar foobar"]
    references = ["hello there general kenobi", "foo bar foobar"]
    assert (
        compute_with_references("exact_match", references, predictions) == 100
    )

    predictions = [
        "It is a guide to action which ensures that the military always obeys the commands of the party"
    ]
    references = [
        "It is a guide to action which ensures that the military always obeys the commands of the party"
    ]
    assert np.isclose(
        compute_with_references("meteor", references, predictions),
        1,
        atol=1e-3,
    )

    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]

    for rouge_variant in ["1", "2", "L", "Lsum"]:
        for rouge_metric in ["precision", "recall", "fmeasure"]:
            assert (
                compute_with_references(
                    f"rouge-{rouge_variant}-{rouge_metric}",
                    references,
                    predictions,
                )
                == 1.0
            )

    assert compute_with_references("rouge", references, predictions) == 1.0

    predictions = ["hello there", "general kenobi"]
    references = ["hello there", "general kenobi"]

    assert (
        compute_with_references(
            "exact_match", references=references, predictions=predictions
        )
        == 100
    )

    predictions = ["hello there general kenobi", "foo bar foobar"]
    references = ["hello there general kenobi", "foo bar foobar"]

    for fm in FUZZ_METRICS:
        assert compute_with_references(fm, predictions, references) == 100

    predictions = ["hello there general kenobi"]
    references = ["hello there general kenobi"]
    assert compute_with_references("levenshtein", references, predictions) == 0

    predictions = ["i love cake", "i hate broccoli"]
    res = compute_without_references("sentiment-analysis", predictions)
    pos = res[0]["score"]
    neg = res[1]["score"]
    assert np.isclose(pos, 1, atol=1e-3)
    assert np.isclose(neg, 1, atol=1e-2)

    predictions = [
        "where is the capital of france? Paris is the capital of France.",
        "where is the capital of france? The sky is blue.",
    ]
    labels = compute_without_references(
        "cross-encoder/qnli-electra-base",
        predictions=predictions,
    )

    scores = [label["score"] for label in labels]
    assert np.isclose(scores[0], 1, atol=1e-2)
    assert np.isclose(scores[1], 0, atol=1e-2)


if __name__ == "__main__":
    run_tests()
