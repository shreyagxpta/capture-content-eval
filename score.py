import json

from evaluate import load_yaml, load_json


def results_by_model(results):
    """Split the flat results list into one list per model name."""
    groups = {}
    for r in results:
        groups.setdefault(r["model"], []).append(r)
    return groups


def primary_accuracy(results):
    """Fraction of items where the model's primary matches the gold primary."""
    correct = 0
    for r in results:
        if r["predicted"]["primary"] == r["gold_primary"]:
            correct += 1
    return correct / len(results)


def label_scores(predicted_labels, gold_labels):
    """Precision, recall, and F1 for one photo's predicted labels against the gold set."""
    predicted = set(predicted_labels)
    gold = set(gold_labels)
    correct = predicted & gold
    precision = len(correct) / len(predicted) if predicted else 0.0
    recall = len(correct) / len(gold) if gold else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def average_label_scores(results):
    """Average precision, recall, and F1 across all items for one model."""
    precisions, recalls, f1s = [], [], []
    for r in results:
        predicted_labels = [s["label"] for s in r["predicted"]["secondary"]]
        precision, recall, f1 = label_scores(predicted_labels, r["gold_secondary"])
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    n = len(results)
    return sum(precisions) / n, sum(recalls) / n, sum(f1s) / n


def reliability_table(results):
    """Group predicted labels into confidence buckets, and report per bucket the
    average stated confidence against the actual hit rate (how often it was right)."""
    buckets = {"low (<0.6)": [], "medium (0.6-0.75)": [], "high (>=0.75)": []}
    for r in results:
        gold = set(r["gold_secondary"])
        for s in r["predicted"]["secondary"]:
            confidence = s["confidence"]
            hit = 1 if s["label"] in gold else 0
            if confidence < 0.6:
                buckets["low (<0.6)"].append((confidence, hit))
            elif confidence < 0.75:
                buckets["medium (0.6-0.75)"].append((confidence, hit))
            else:
                buckets["high (>=0.75)"].append((confidence, hit))

    rows = []
    for name, entries in buckets.items():
        if not entries:
            continue
        stated = sum(c for c, h in entries) / len(entries)
        actual = sum(h for c, h in entries) / len(entries)
        rows.append((name, len(entries), stated, actual))
    return rows


def brier_score(results):
    """Mean squared gap between a predicted label's confidence and whether it was
    actually right (1 if in gold, 0 if not). Lower is better; 0 is perfect."""
    squared_errors = []
    for r in results:
        gold = set(r["gold_secondary"])
        for s in r["predicted"]["secondary"]:
            outcome = 1 if s["label"] in gold else 0
            squared_errors.append((s["confidence"] - outcome) ** 2)
    return sum(squared_errors) / len(squared_errors) if squared_errors else 0.0


def failure_profile(results, confident_threshold=0.75):
    """Count each failure type for one model across all photos."""
    wrong_primary = 0
    misses = 0
    over_claims = 0
    confident_over_claims = 0
    for r in results:
        gold = set(r["gold_secondary"])
        predicted = {s["label"] for s in r["predicted"]["secondary"]}
        if r["predicted"]["primary"] != r["gold_primary"]:
            wrong_primary += 1
        misses += len(gold - predicted)
        for s in r["predicted"]["secondary"]:
            if s["label"] not in gold:
                over_claims += 1
                if s["confidence"] >= confident_threshold:
                    confident_over_claims += 1
    return {
        "wrong_primary": wrong_primary,
        "misses": misses,
        "over_claims": over_claims,
        "confident_over_claims": confident_over_claims,
    }


def takeaway_line(groups, profiles):
    """Build a one-line summary naming each model's dominant failure."""
    names = list(groups)
    if len(names) != 2:
        return "See the failure profile above for how each model fails."

    def describe(name):
        p = profiles[name]
        brier = brier_score(groups[name])
        # the dominant failure is whichever count is highest
        worst_type, worst_count = max(
            [("over-claiming", p["over_claims"]),
             ("missing real labels", p["misses"]),
             ("wrong primaries", p["wrong_primary"])],
            key=lambda pair: pair[1],
        )
        if worst_count == 0:
            return f"{name} makes almost no errors (Brier {brier:.3f})"
        note = f"{worst_count} {worst_type}"
        if worst_type == "over-claiming" and p["confident_over_claims"] > 0:
            note += f", {p['confident_over_claims']} of them confident"
        return f"{name}'s main weakness is {note} (Brier {brier:.3f})"

    a, b = names
    fa = average_label_scores(groups[a])[2]
    fb = average_label_scores(groups[b])[2]
    return (
        f"Both models score high (F1 {fa:.0%} for {a}, {fb:.0%} for {b}), but trade off differently: "
        f"{describe(a)}; {describe(b)}. Lower Brier means confidence you can trust unattended."
    )


def write_report(results, path):
    """Write a human-readable markdown report, generated straight from the results."""
    groups = results_by_model(results)
    profiles = {name: failure_profile(res) for name, res in groups.items()}
    names = list(groups)

    lines = []
    lines.append("# capture-content-eval results")
    lines.append("")
    lines.append(
        "Frontier models label capture metadata from a text description: a primary type "
        "(subject vs scene) plus secondary labels with confidence. Scored on primary accuracy, "
        "label precision / recall / F1, and confidence calibration (Brier, lower is better)."
    )
    lines.append("")
    lines.append("## Scores")
    lines.append("")
    lines.append("| model | primary acc | precision | recall | F1 | Brier |")
    lines.append("|---|---|---|---|---|---|")
    for name in names:
        res = groups[name]
        acc = primary_accuracy(res)
        p, r, f1 = average_label_scores(res)
        lines.append(f"| {name} | {acc:.0%} | {p:.0%} | {r:.0%} | {f1:.0%} | {brier_score(res):.3f} |")
    lines.append("")
    lines.append("## Failure profile")
    lines.append("")
    lines.append("| failure type | " + " | ".join(names) + " |")
    lines.append("|" + "---|" * (len(names) + 1))
    for ft in ["wrong_primary", "misses", "over_claims", "confident_over_claims"]:
        lines.append(f"| {ft} | " + " | ".join(str(profiles[n][ft]) for n in names) + " |")
    lines.append("")
    lines.append("## Takeaway")
    lines.append("")
    lines.append(takeaway_line(groups, profiles))
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    config = load_yaml("config.yaml")
    results = load_json(config["paths"]["results"])

    for model_name, model_results in results_by_model(results).items():
        accuracy = primary_accuracy(model_results)
        hits = round(accuracy * len(model_results))
        missed = [r["id"] for r in model_results if r["predicted"]["primary"] != r["gold_primary"]]
        print(f"{model_name}: primary accuracy = {accuracy:.0%}  ({hits}/{len(model_results)})")
        print(f"   missed primary on: {missed}")

        precision, recall, f1 = average_label_scores(model_results)
        print(f"   labels: precision = {precision:.0%}, recall = {recall:.0%}, F1 = {f1:.0%}")

        print("   calibration (stated confidence vs actual hit rate):")
        for name, count, stated, actual in reliability_table(model_results):
            print(f"      {name:<18} n={count:<3} stated={stated:.0%}  actual={actual:.0%}")
        print(f"   calibration score (Brier, lower is better): {brier_score(model_results):.3f}")

    print()
    print("=== failure profile (counts across all photos) ===")
    profiles = {name: failure_profile(res) for name, res in results_by_model(results).items()}
    names = list(profiles)
    print("   " + "failure type".ljust(26) + "".join(n.rjust(10) for n in names))
    for failure_type in ["wrong_primary", "misses", "over_claims", "confident_over_claims"]:
        counts = "".join(str(profiles[n][failure_type]).rjust(10) for n in names)
        print("   " + failure_type.ljust(26) + counts)

    write_report(results, config["paths"]["report"])
    print(f"\nwrote report to {config['paths']['report']}")
