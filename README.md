<br /><br />
<div align="center">
  <h1 align="center">text-compress-scoring</h1>
  <h4 align="center">A Python SDK for scoring text compression.</h4>
</div>

## Installation

```bash
pip install git+https://github.com/condenses/text-compress-scoring.git
```

## Configuration (Environment Variables)

```bash
# Reward Model Configuration
REWARD_MODEL_CONFIG__MODEL_NAME=Skywork/Skywork-Reward-Gemma-2-27B-v0.2
REWARD_MODEL_CONFIG__TEMPERATURE=1.5
REWARD_MODEL_CONFIG__COMPRESSION_SCALE=0.2
REWARD_MODEL_CONFIG__TIKTOKEN_MODEL=gpt-4o
REWARD_MODEL_CONFIG__REFERENCE_SCORE=0.5

# Prompt Guard Configuration
PROMPT_GUARD_CONFIG__MODEL_NAME=katanemo/Arch-Guard
PROMPT_GUARD_CONFIG__DEVICE=cuda

# Scoring Client Configuration
SCORING_CLIENT_CONFIG__HOST=0.0.0.0
SCORING_CLIENT_CONFIG__PORT=9102
SCORING_CLIENT_CONFIG__TIMEOUT=32.0
```

## Usage

### Scoring Messages

```python
from text_compress_scoring.schemas import Message, BatchScoringRequest
from text_compress_scoring.scoring_modeling import ScoringModel

# Initialize the scoring model
model = ScoringModel()

# Create messages
original_messages = [
    Message(role="user", content="Original message", is_compressed=False)
]

compressed_messages = [
    Message(role="user", content="Compressed version", is_compressed=True)
]

# Score single messages
score = model.score_messages(original_messages)

# Create batch request
request = BatchScoringRequest(
    batch_compressed_messages=[compressed_messages],
    original_messages=original_messages
)
```

### Using the Server

```python
from text_compress_scoring.server import app
import uvicorn

# Start the server
uvicorn.run(app, host="0.0.0.0", port=9101)
```

## API Reference

### Message Model

```python
class Message:
    role: str           # Role of the message sender (e.g., "user")
    content: str        # Content of the message
    is_compressed: bool # Whether this is a compressed message
```

### Scoring Endpoints

#### POST /api/scoring
Score a batch of compressed messages against original messages.

Request body:
```python
class BatchScoringRequest:
    batch_compressed_messages: List[List[Message]]
    original_messages: List[Message]
```

Response:
```python
class BatchScoringResponse:
    scores: List[float]
```

## Error Handling

The SDK includes built-in error handling for model operations and API requests:

```python
try:
    score = model.score_messages(messages)
except Exception as e:
    print(f"Scoring failed: {e}")
```

## Dependencies

- PyTorch
- Transformers
- FastAPI
- Pydantic
- Loguru

For a complete list of dependencies, please refer to the pyproject.toml file.
