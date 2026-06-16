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
    """
    examples_text = ""

    for example in labeled_examples:
        examples_text += (
            f"Title: {example['title']}\n"
            f"Podcast: {example['podcast']}\n"
            f"Description: {example['description']}\n"
            f"Label: {example['label']}\n\n"
        )

    prompt = f"""
You are a podcast episode format classifier.

Your job is to classify a podcast episode description into exactly one of these labels:

- interview: one main guest is being interviewed or is the focus of a conversation
- solo: one host speaks from personal experience, opinion, memory, or reflection
- panel: three or more people discuss, debate, or compare perspectives
- narrative: a reported or produced story built from events, sources, records, interviews, or a clear story arc

Use the labeled examples below to learn the pattern.

Labeled examples:
{examples_text}

Now classify this new episode description:

Description: {description}

Return your answer in exactly this format:
Label: <one of interview, solo, panel, narrative>
Reasoning: <one brief sentence explaining why>
"""
    return prompt


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.
    """
    try:
        prompt = build_few_shot_prompt(labeled_examples, description)

        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful podcast format classifier. Follow the requested output format exactly.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
        )

        response_text = response.choices[0].message.content.strip()

        label = "unknown"
        reasoning = response_text

        for line in response_text.splitlines():
            clean_line = line.strip()

            if clean_line.lower().startswith("label:"):
                parsed_label = clean_line.split(":", 1)[1].strip().lower()
                parsed_label = (
                    parsed_label
                    .replace("*", "")
                    .replace(".", "")
                    .replace(",", "")
                    .strip()
                )

                if parsed_label in VALID_LABELS:
                    label = parsed_label

            elif clean_line.lower().startswith("reasoning:"):
                reasoning = clean_line.split(":", 1)[1].strip()

        return {
            "label": label,
            "reasoning": reasoning,
        }

    except Exception as e:
        return {
            "label": "unknown",
            "reasoning": f"Classifier error: {e}",
        }