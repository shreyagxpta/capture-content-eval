import json
import sys
import time

import yaml


def load_yaml(path):
    """Read a YAML file, or stop with a clear message if it is missing or broken."""
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        sys.exit(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        sys.exit(f"Config file {path} is not valid YAML: {e}")


def load_json(path):
    """Read a JSON file, or stop with a clear message if it is missing or broken."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        sys.exit(f"Benchmark file not found: {path}")
    except json.JSONDecodeError as e:
        sys.exit(f"Benchmark file {path} is not valid JSON: {e}")


def build_prompt(description, label_menu):
    """Assemble the instruction sent to a model for one photo description."""
    menu = ", ".join(label_menu)
    return f"""You are labeling a photo from a text description. You cannot see the image, only the text.

Description: "{description}"

1. PRIMARY: Decide what the photo is mainly about by what the eye locks onto first.
   - "subject" if one element clearly stands out as the focus.
   - "scene" if attention is spread with no single focus.
   Size only breaks ties: a small element can still be the focus.

2. SECONDARY: From this fixed menu, list every label the description states or forces:
   {menu}
   Only include a label the description says or makes unavoidable. Add a label
   when the words force it: "sunrise" forces "sky", because you cannot see a
   sunrise without sky. But do not add a label just because it usually goes with
   the scene: do not add "outdoors" for a sport unless the description says where
   it happens.

3. CONFIDENCE: Give each secondary label a confidence from 0 to 1.

Return only JSON in exactly this shape, with no other text:
{{"primary": "subject", "secondary": [{{"label": "animal", "confidence": 0.9}}]}}
"""


def call_mock(prompt):
    """Stand-in for a real model: returns a canned answer so the pipeline runs
    with no API key and no cost, for testing before the live run."""
    return (
        '{"primary": "subject", '
        '"secondary": ['
        '{"label": "people", "confidence": 0.8}, '
        '{"label": "sports", "confidence": 0.7}, '
        '{"label": "outdoors", "confidence": 0.55}]}'
    )


def call_openai(model, prompt, generation, timeout):
    """Send the prompt to an OpenAI model. Reads OPENAI_API_KEY from the environment.
    GPT-5 and o-series are reasoning models: they reject 'temperature' and use
    'max_completion_tokens', so branch on the model name."""
    from openai import OpenAI
    client = OpenAI(timeout=timeout)
    params = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if model.startswith("gpt-5") or model.startswith("o"):
        params["max_completion_tokens"] = max(generation["max_tokens"], 2000)
        params["reasoning_effort"] = "low"
    else:
        params["temperature"] = generation["temperature"]
        params["max_tokens"] = generation["max_tokens"]
    response = client.chat.completions.create(**params)
    return response.choices[0].message.content


def call_anthropic(model, prompt, generation, timeout):
    """Send the prompt to an Anthropic model. Reads ANTHROPIC_API_KEY from the environment."""
    from anthropic import Anthropic
    client = Anthropic(timeout=timeout)
    response = client.messages.create(
        model=model,
        max_tokens=generation["max_tokens"],
        temperature=generation["temperature"],
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def ask_model(model_cfg, prompt, generation, timeout, use_mock=False):
    """Send one prompt to the model in model_cfg and return its raw text answer.
    Routes to the right provider, or the mock stand-in when use_mock is True."""
    if use_mock:
        return call_mock(prompt)
    provider = model_cfg["provider"]
    if provider == "openai":
        return call_openai(model_cfg["model"], prompt, generation, timeout)
    if provider == "anthropic":
        return call_anthropic(model_cfg["model"], prompt, generation, timeout)
    sys.exit(f"Unknown provider in config: {provider}")


def ask_model_safely(model_cfg, prompt, generation, reliability, use_mock=False):
    """Call the model, retrying with a growing wait if it fails, up to max_retries."""
    delay = reliability["backoff_seconds"]
    for attempt in range(1, reliability["max_retries"] + 1):
        try:
            return ask_model(model_cfg, prompt, generation, reliability["timeout_seconds"], use_mock)
        except Exception as error:
            if attempt == reliability["max_retries"]:
                raise
            print(f"  attempt {attempt} failed ({error}); retrying in {delay}s")
            time.sleep(delay)
            delay *= 2


def extract_json(raw):
    """Pull the JSON object out of a reply, ignoring any chatter or code fences around it."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return raw  # no braces found; let the parse step fail and flag it
    return raw[start:end + 1]


def parse_answer(raw):
    """Read a model reply into a dict. Strips chatter, and flags a parse failure
    instead of crashing if the text is not valid JSON."""
    text = extract_json(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"ok": False, "primary": None, "secondary": [], "raw": raw}
    return {
        "ok": True,
        "primary": data.get("primary"),
        "secondary": data.get("secondary", []),
        "raw": raw,
    }


def evaluate_item(model_cfg, item, generation, reliability, label_menu, use_mock=False):
    """Run one photo through one model. Catches a total call failure so one bad
    item cannot sink the whole run, and records the result either way."""
    prompt = build_prompt(item["description"], label_menu)
    try:
        raw = ask_model_safely(model_cfg, prompt, generation, reliability, use_mock)
        predicted = parse_answer(raw)
    except Exception as error:
        predicted = {"ok": False, "primary": None, "secondary": [], "raw": f"call failed: {error}"}
    return {
        "model": model_cfg["name"],
        "id": item["id"],
        "gold_primary": item["primary"],
        "gold_secondary": item["secondary"],
        "predicted": predicted,
    }


def run_eval(config, benchmark, use_mock=False):
    """Run every model over every benchmark item and collect all the results."""
    results = []
    for model_cfg in config["models"]:
        print(f"Running model: {model_cfg['name']}")
        for item in benchmark["items"]:
            result = evaluate_item(
                model_cfg,
                item,
                config["generation"],
                config["reliability"],
                benchmark["label_menu"],
                use_mock,
            )
            status = "ok" if result["predicted"]["ok"] else "FAILED"
            print(f"  {item['id']}: {status}")
            results.append(result)
    return results


if __name__ == "__main__":
    config = load_yaml("config.yaml")
    benchmark = load_json(config["paths"]["benchmark"])

    use_mock = "--real" not in sys.argv  # plain run uses the mock; add --real to call live models
    results = run_eval(config, benchmark, use_mock)

    with open(config["paths"]["results"], "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} results to {config['paths']['results']}")
