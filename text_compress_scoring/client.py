from typing import List
import httpx
from .schemas import BatchScoringRequest, BatchScoringResponse
from .config import CONFIG


class ScoringClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=CONFIG.scoring_client_config.timeout)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()

    def score_batch(
        self,
        original_user_message: str,
        batch_compressed_user_messages: List[str],
        timeout: float = CONFIG.scoring_client_config.timeout,
    ) -> List[float]:
        """Score a batch of compressed messages against original message.

        Args:
            original_user_message: The original user message to compare against
            batch_compressed_user_messages: List of compressed messages to score

        Returns:
            List of scores between 0 and 1, where higher scores indicate better compression.
            A score of 0 indicates the message failed prompt guard checks.
        """
        request = BatchScoringRequest(
            original_user_message=original_user_message,
            batch_compressed_user_messages=batch_compressed_user_messages,
        )
        response = self.client.post(
            f"{self.base_url}/api/scoring",
            json=request.model_dump(),
            timeout=timeout,
        )
        response.raise_for_status()
        return BatchScoringResponse(**response.json()).scores


class AsyncScoringClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=CONFIG.scoring_client_config.timeout)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def score_batch(
        self,
        original_user_message: str,
        batch_compressed_user_messages: List[str],
        timeout: float = CONFIG.scoring_client_config.timeout,
    ) -> List[float]:
        """Score a batch of compressed messages against original message.

        Args:
            original_user_message: The original user message to compare against
            batch_compressed_user_messages: List of compressed messages to score

        Returns:
            List of scores between 0 and 1, where higher scores indicate better compression.
            A score of 0 indicates the message failed prompt guard checks.
        """
        request = BatchScoringRequest(
            original_user_message=original_user_message,
            batch_compressed_user_messages=batch_compressed_user_messages,
        )
        response = await self.client.post(
            f"{self.base_url}/api/scoring",
            json=request.model_dump(),
            timeout=timeout,
        )
        response.raise_for_status()
        return BatchScoringResponse(**response.json()).scores


# Usage examples:
if __name__ == "__main__":
    # Example messages
    original_msg = "Hello, how are you?"
    compressed_msgs = ["Hi", "Hello"]

    # Synchronous usage
    with ScoringClient(CONFIG.scoring_client_config.base_url) as client:
        scores = client.score_batch(original_msg, compressed_msgs)
        print(f"Compression scores: {scores}")

    # Async usage
    import asyncio

    async def main():
        async with AsyncScoringClient(CONFIG.scoring_client_config.base_url) as client:
            scores = await client.score_batch(original_msg, compressed_msgs)
            print(f"Compression scores: {scores}")

    # Run async example
    asyncio.run(main())
