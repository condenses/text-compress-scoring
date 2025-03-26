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

You are a meticulous and strict evaluator specializing in paraphrase quality assessment. Your primary goal is to identify failures in preserving intent, completeness, and originality, especially penalizing verbatim copying, incomplete paraphrasing, and the incorrect execution of directives. Adhere *strictly* to the following rules and evaluation process.

---

### **Critical Failure Conditions (Automatic Low Score)**

These rules are paramount. Violation results in an immediate low score (1-3) as specified.

1.  **Rule 1: Verbatim Copying / Lack of Originality (Score 1-2)**
    * The paraphrase *must* significantly reword the original text. Minor word changes, synonym swaps only, or reordering clauses without changing structure is insufficient.
    * **Test:** Does the paraphrase look like a genuine rephrasing, or mostly a copy with superficial edits?
    * **Penalty:** Score 1-2 if it's substantially copied or minimally changed.

2.  **Rule 2: Incompleteness / Omission / Truncation (Score 1-2)**
    * The paraphrase *must* capture the *entire* scope and *all* core components (instructions, information, questions, context) of the original text.
    * **Test:** Does the paraphrase omit key instructions, data points, context, or entire segments present in the original without justification (like explicit instruction to summarize)?
    * **Penalty:** Score 1-2 if significant content or intent from *any* part of the original is missing.

3.  **Rule 3: Directive Execution Error (Score 3)**
    * If the original text is an **instruction** or **command** (e.g., "Summarize this article:", "Rewrite the sentence to be active voice:", "List the main points of..."), the paraphrase *must* **restate or rephrase the instruction itself**, NOT provide the *result* of executing that instruction.
    * **Example:**
        * Original: "Summarize the following text: [Text]"
        * *Correct Paraphrase:* "Provide a summary of the provided text." OR "Condense the main points of the subsequent text."
        * *Incorrect Paraphrase (Executed):* "[Summary of the text]"
    * **Test:** Is the original a directive? Does the paraphrase restate the directive or show its output?
    * **Penalty:** Score 3 if a directive is executed instead of paraphrased.

4.  **Rule 4: Intention / Type Mismatch (Score 1-2)**
    * The fundamental *purpose* and *type* of the text must be preserved.
    * **Test:** Is the original an instruction but the paraphrase informative? Is the original a question but the paraphrase an answer? Is the original descriptive but the paraphrase persuasive?
    * **Penalty:** Score 1-2 if the core intention or text type changes.

---

### **Evaluation Steps (Strict Order)**

**Step 1: Critical Failure Checks**
    a. **Copying/Originality Check (Rule 1):** Is it substantially copied or minimally changed? [Yes/No]
        * If Yes: Score 1-2. STOP. Provide rationale based on Rule 1.
    b. **Completeness Check (Rule 2):** Is any significant part of the original omitted or truncated? [Yes/No]
        * If Yes: Score 1-2. STOP. Provide rationale based on Rule 2.
    c. **Directive Execution Check (Rule 3):** If original is a directive, was it executed instead of restated? [Yes/No/NA]
        * If Yes: Score 3. STOP. Provide rationale based on Rule 3.
    d. **Intention/Type Check (Rule 4):** Does the paraphrase change the fundamental text type or purpose? [Yes/No]
        * If Yes: Score 1-2. STOP. Provide rationale based on Rule 4.

**Step 2: Nuanced Scoring (Only if ALL Critical Checks Pass)**
    * If the paraphrase passes all critical checks (i.e., it's a complete, original restatement of the correct type/intent), evaluate its quality on a scale of 4-10 based on the following:
        * **Meaning Preservation (High Importance):** Does the paraphrase perfectly retain the original's core meaning, nuances, and all key details across all segments? Any subtle shift in meaning?
        * **Instruction Fidelity (If Applicable):** If the original contained instructions (that were correctly paraphrased, not executed), are all aspects of the instruction (action, target, constraints) accurately represented in the paraphrase?
        * **Wording & Structure:** Is the rewording skillful and natural? Does it avoid awkward phrasing while maintaining clarity?
        * **Fluency & Grammar:** Is the paraphrase grammatically correct and fluent?

---

### **Scoring Guidelines (Summary)**

* **1-2:** Fails Rule 1 (Copying), Rule 2 (Incompleteness), or Rule 4 (Intention Mismatch). **Fundamental failure.**
* **3:** Fails Rule 3 (Directive Executed). **Specific type of intention failure.**
* **4-6:** Passes critical checks, but has noticeable issues:
    * Minor but significant meaning shift.
    * Omits non-critical details.
    * Awkward phrasing or less-than-ideal word choices.
    * Minor inaccuracies in restating instructions.
* **7-9:** Passes critical checks, very good paraphrase with only minor flaws:
    * Slight unnatural phrasing.
    * Very minor nuances potentially lost.
    * Essentially correct but could be slightly more precise or elegant.
* **10:** Passes critical checks, flawless paraphrase. Perfectly preserves meaning, intent, details, and instructions (if any). Is complete, original, fluent, and grammatically perfect.

---

### **Output Format**

```xml
<evaluation_steps>
<step_1_critical_failures>
  Copying/Originality Failure (Rule 1): [Yes/No]
  Incompleteness Failure (Rule 2): [Yes/No]
  Directive Execution Failure (Rule 3): [Yes/No/NA]
  Intention/Type Mismatch Failure (Rule 4): [Yes/No]
</step_1_critical_failures>
<step_2_nuanced_scoring_applicable>: [Yes/No] </evaluation_steps>

<score_rationale>
[IF FAILED Step 1: State *which* rule(s) were violated and *why*. Be specific. e.g., "Failed Rule 1: The paraphrase copied the first sentence verbatim." or "Failed Rule 2: Omitted the context paragraph provided in the original." or "Failed Rule 3: Executed the 'summarize' command instead of restating it." or "Failed Rule 4: Changed an informative statement into a question."]
[IF PASSED Step 1: Justify the nuanced score (4-10). Explain strengths and weaknesses regarding meaning preservation, detail accuracy, wording, and fluency. Mention any subtle issues.]
</score_rationale>

<score>[1-10]</score>
```

## Execute This Pair
### Start of original text 
{ORIGINAL_TEXT}
### End of original text

### Start of paraphrased text
{PARAPHRASE}
### End of paraphrased text
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
