import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.

    TODO — Milestone 2:

    Your prompt needs to:
      1. Describe the task and the four valid labels
      2. Show the labeled training examples so the LLM can learn the pattern
      3. Present the new description and ask for a classification

    The LLM should return a single label from VALID_LABELS (exactly as written)
    plus a brief explanation of its reasoning. Think carefully about the output
    format you request — you'll need to parse it in classify_episode().

    Before writing code, complete specs/classifier-spec.md.
    """
    # 1. Task instruction: describe the task and the four valid labels
    task_instruction = (
        "You are classifying podcast episodes by their format. "
        "Classify the episode into exactly one of these four labels:\n\n"
        "- interview: a conversation between a host and one or more guests\n"
        "- solo: a single host speaking from memory, experience, or opinion — no guests, "
        "no assembled external sources\n"
        "- panel: multiple guests with roughly equal speaking time, often debating or "
        "discussing a topic together\n"
        "- narrative: a story assembled from external sources — interviews, archival "
        "audio, reporting — with a clear narrative arc\n\n"
        "Return the label on the first line, then your reasoning on the following lines. "
        "The label must be exactly one of: interview, solo, panel, narrative."
    )

    # 2. Labeled examples block: Title / Description / Label for each training example
    examples_block = ""
    for ex in labeled_examples:
        examples_block += (
            f"Title: {ex['title']}\n"
            f"Description: {ex['description']}\n"
            f"Label: {ex['label']}\n\n"
            "---\n\n"
        )

    # 3. New episode: same format as examples but Label: ?, with classification instruction
    new_episode = (
        f"Title: (unknown)\n"
        f"Description: {description}\n"
        f"Label: ?\n\n"
        "Classify the episode above. Return the label on the first line, "
        "then your reasoning below it."
    )

    return f"{task_instruction}\n\n{examples_block}{new_episode}"


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.
    Returns a dict with "label" (one of VALID_LABELS or "unknown") and "reasoning".
    """
    try:
        # Step 1: Build the prompt using labeled training examples
        prompt = build_few_shot_prompt(labeled_examples, description)

        # Step 2: Send the prompt to the LLM
        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
        )

        # Step 3: Parse the response — label on first line, reasoning on the rest
        raw = response.choices[0].message.content.strip()
        lines = raw.splitlines()
        first_line = lines[0].strip().lower() if lines else ""
        reasoning = "\n".join(lines[1:]).strip() if len(lines) > 1 else raw

        # Step 4: Validate the label — scan first line for a known label, else "unknown"
        label = next((l for l in VALID_LABELS if l in first_line), "unknown")

        # Step 5: Return the result dict
        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        return {"label": "unknown", "reasoning": f"Error: {e}"}
