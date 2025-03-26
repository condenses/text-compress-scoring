import numpy as np
from loguru import logger
from pydantic import BaseModel
from openai import OpenAI
import re
from transformers import pipeline
from .config import CONFIG
import torch
from .utils import retry

PARAPHRASE_SCORE_PROMPT = """
### **Role:**

You are an expert evaluator trained to detect subtle intention mismatches and incomplete paraphrasing, especially when directives (commands) are incorrectly transformed into executed outputs or when portions of the original text are omitted.

---

### **Core Principles**

1.  **Directive Preservation Rule (Instructions MUST be Restated):**

    *   If the original text is an **instruction** (e.g., "Summarize the following paragraph..."), the paraphrase *must* **restate the instruction**, NOT show the *executed result* of that instruction.
    *   *Automatic penalty*: Score ≤ 3 if the paraphrase *executes* the command instead of restating it.

2.  **Intention Hierarchy:**

    *   **Primary Focus:** Compare *what the texts are trying to achieve* (goal/purpose), not just content. The overall *intent* must be identical.
    *   **Type Alignment:** Original and paraphrase must share the same *text type* (directive, informative, persuasive, etc.). A directive must remain a directive; an informative text must remain informative.

3.  **True Paraphrase Requirement (All Text Segments):**

    *   *Automatic penalty*: Score ≤ 2 if the paraphrase merely copies the original text verbatim or with minimal changes (less than 70% substantially reworded).
    *   A genuine paraphrase must substantially rework wording *while preserving meaning*. At least 70% of the text should use different words, phrases, or sentence structures.
    *   **All Text Segments Rule:** If the original text comprises multiple segments (e.g., instructions + an excerpt, question + context), *each segment* must be independently and completely paraphrased. Failure to paraphrase *any* segment constitutes a failure of the true paraphrase test. No portion of the original should be directly copied (unless quotation is explicitly permitted and properly attributed, and even then, a paraphrase is preferred).

    * **Completeness:** The paraphrase must represent a complete rephrasing of the *entire* original text's intention and *all* its content. Unjustified truncation, omission, or summarization of core instructional elements or key concepts/text segments is heavily penalized, even if the remaining portion is well-paraphrased. (See "Meaning & Instruction Adherence" below).

---

### **Evaluation Steps**

#### **1. Paraphrase Assessment (Mandatory First Step)**

```plaintext
a. Is this a true paraphrase (considering all text segments)? [Yes/No]
b. Estimated percentage of text substantially reworded (across all segments): [%]
c. Passes minimum paraphrase threshold (>70%)? [Yes/No]

If fails true paraphrase test: Score 1-2 and halt.
```

#### **2. Intention Analysis (Second Mandatory Step)**

```plaintext
a. Original Text Type: [Directive/Informative/Persuasive/Query]
b. Paraphrase Text Type: [Directive/Informative/Persuasive/Query]
c. Type Match? [Yes/No]
d. For Directives:
    - Preserved Action? (e.g., "organize" → "categorize") [Yes/No]
    - Preserved Target? (e.g., "findings" → "results") [Yes/No]
    - Preserved Output Format? (e.g., "four categories") [Yes/No]
e. For Informative:
    - Same perspective (1st/3rd person)? [Yes/No]
    - Same information purpose (summary/analysis)? [Yes/No]

If types mismatch (e.g., directive → informative): Score 1-2 and halt.
```

#### **3. Conditional Scoring**

Only proceed if intentions align (same type + same goal) *and* the paraphrase assessment passes.

**Criteria (Weighted):**

1.  **Meaning & Instruction Adherence (40%)**

    *   Does the paraphrase fully retain the original’s core meaning *across all text segments*?
    *   For directives: Does it restate *all* instructions exactly?
    *   **Completeness Check:** Does the paraphrase address the *full scope* of the original's intent *and* include a full paraphrase of *all text segments*? Are there any significant omissions, truncations, or unjustified summaries that alter the core meaning, instructions, or content, even if the rephrased parts are accurate? If significant portions are missing without justification, heavily penalize here.

2.  **Key Details (30%)**

    *   Are all critical facts, entities, and nuances preserved *across all text segments*?

3.  **Wording & Structure (20%)**

    *   Significant rephrasing without plagiarism *across all text segments*.

4.  **Grammar & Fluency (10%)**

    *   Natural, error-free language.

---

### **Scoring Guidelines**

*   **1-2:** Fundamental intention mismatch (e.g., command → output) OR failure of the true paraphrase test (including failure to paraphrase all segments).
*   **3:** The paraphrase executes the command instead of restating.
*   **4-6:** Partial alignment (minor goal drift, omissions, or incomplete paraphrasing of some segments).
*   **7-9:** Near-perfect alignment (minor phrasing issues).
*   **10:** Flawless (identical goals, perfect execution, all segments fully paraphrased).

---

### **Output Format**

```
<paraphrase_assessment>
True Paraphrase: [Yes/No]
Paraphrased text is truncated or incompleted: [Yes/No]
Passes Threshold: [Yes/No]
</paraphrase_assessment>

<intention_analysis>
Original Type: [Type]
Paraphrase Type: [Type]
Type Match: [Yes/No]
Core Action Preserved? [Yes/No] (if directive)
Perspective Match? [Yes/No] (if informative)
</intention_analysis>

<score_rationale>
[Note if text was copied rather than paraphrased, specifically mentioning which segment(s) were copied.]
[Explicitly note if paraphrase executed instead of restated commands.]
[Detail any intention discrepancies.]
[Specifically address any unjustified truncations, omissions, or summarizations, referencing the "Completeness Check" and the "All Text Segments Rule."]
[Identify which segments, if any, were not fully paraphrased.]
</score_rationale>

<score>[1-10]</score>
```
----
### Execute This Pair
#### Original Text
{ORIGINAL_TEXT}
---
#### Paraphrased Text
{PARAPHRASE}
"""


class ParaphraseScorer:
    def __init__(self, openai_client: OpenAI):
        self.llm_client = openai_client
        self.model = CONFIG.vllm_config.model_name
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    @retry(max_retries=3, retry_delay=5)
    def single_paraphrase_grade(
        self, original_text: str, paraphrase: str
    ) -> tuple[str, int]:
        """
        Compute the paraphrase score between original text and its paraphrase.
        Returns a tuple of (feedback, score) where score is between 1 and 10.
        """
        prompt = PARAPHRASE_SCORE_PROMPT.format(
            ORIGINAL_TEXT=original_text,
            PARAPHRASE=paraphrase,
        )
        response = self.llm_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=CONFIG.vllm_config.temperature,
        )
        completion = response.choices[0].message.content

        # Fix regex to handle both spellings of explanation
        score_match = re.search(r"<score>\s*(\d+)\s*</score>", completion, re.DOTALL)
        explanation_match = re.search(
            r"<intention_analysis>(.*?)</intention_analysis>",
            completion,
            re.DOTALL,
        )

        rationale_match = re.search(
            r"<score_rationale>(.*?)</score_rationale>",
            completion,
            re.DOTALL,
        )

        if not score_match:
            logger.warning(f"Could not parse completion: {completion}")
            return "No feedback provided", 1

        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        logger.info(
            f"Paraphrase scoring: Original: {original_text[:32]}... | Paraphrase: {paraphrase[:32]}... | {response.usage.prompt_tokens} input tokens | {response.usage.completion_tokens} output tokens"
        )

        score = score_match.group(1).strip()

        # Extract explanation if available
        explanation = ""
        if explanation_match:
            explanation += explanation_match.group(1).strip() + "\n\n"

        if rationale_match:
            explanation += rationale_match.group(1).strip()

        if not explanation:
            explanation = "No explanation provided"

        return explanation, int(score)

    def score_paraphrase_batch(
        self, original_message: str, compressed_messages: list[str]
    ) -> list[float]:
        """
        Compute paraphrase scores for a batch of original texts and their paraphrases.
        Returns normalized scores between 0 and 1.
        """
        scores = []
        for compressed_message in compressed_messages:
            feedback, score = self.single_paraphrase_grade(
                original_message, compressed_message
            )
            logger.info(f"Paraphrase feedback: {feedback} | score: {score}")
            scores.append(score)

        # Normalize scores from 1-10 to 0-1
        normalized_scores = [min(1, score / 10.0) for score in scores]
        return normalized_scores


class GuardingModel:
    def __init__(self):
        self.prompt_guard = pipeline(
            "text-classification",
            model=CONFIG.prompt_guard_config.model_name,
            device=CONFIG.prompt_guard_config.device,  # Use 'device' if supported
        )

    @torch.no_grad()
    def guard(self, prompt: str) -> bool:
        """
        Evaluate a prompt with the text-classification pipeline.
        Returns True if the prompt is classified as a "JAILBREAK".
        """
        try:
            result = self.prompt_guard(prompt)
            logger.info(f"Prompt guard result: {result} | prompt: {prompt[:32]}...")
            return result[0]["label"] == "JAILBREAK"
        except Exception as e:
            logger.error(f"Error in prompt guard: {e}")
            return True
