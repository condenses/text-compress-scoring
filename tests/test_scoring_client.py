import pytest
import asyncio
from text_compress_scoring.client import ScoringClient, AsyncScoringClient
from text_compress_scoring.config import CONFIG

# Test data
TEST_MESSAGES = {
    "original": "Please explain quantum computing in simple terms.",
    "compressed": [
        "Explain quantum computing simply",
        "What is quantum computing?",
        "Quantum computing basics",
    ],
}


def test_sync_scoring_client():
    """Test synchronous scoring client"""
    with ScoringClient(
        f"http://{CONFIG.scoring_client_config.host}:{CONFIG.scoring_client_config.port}"
    ) as client:
        scores = client.score_batch(
            TEST_MESSAGES["original"], TEST_MESSAGES["compressed"]
        )

        # Basic validation of returned scores
        assert isinstance(scores, list)
        assert len(scores) == len(TEST_MESSAGES["compressed"])
        assert all(isinstance(score, float) for score in scores)
        assert all(0 <= score <= 1 for score in scores)


@pytest.mark.asyncio
async def test_async_scoring_client():
    """Test asynchronous scoring client"""
    async with AsyncScoringClient(
        f"http://{CONFIG.scoring_client_config.host}:{CONFIG.scoring_client_config.port}"
    ) as client:
        scores = await client.score_batch(
            TEST_MESSAGES["original"], TEST_MESSAGES["compressed"]
        )

        # Basic validation of returned scores
        assert isinstance(scores, list)
        assert len(scores) == len(TEST_MESSAGES["compressed"])
        assert all(isinstance(score, float) for score in scores)
        assert all(0 <= score <= 1 for score in scores)


@pytest.mark.asyncio
async def test_async_scoring_client_multiple_requests():
    """Test multiple concurrent requests with async client"""
    async with AsyncScoringClient(CONFIG.scoring_client_config.host) as client:
        # Create multiple concurrent requests
        tasks = [
            client.score_batch(TEST_MESSAGES["original"], TEST_MESSAGES["compressed"])
            for _ in range(3)
        ]

        # Run requests concurrently
        results = await asyncio.gather(*tasks)

        # Validate all results
        for scores in results:
            assert isinstance(scores, list)
            assert len(scores) == len(TEST_MESSAGES["compressed"])
            assert all(isinstance(score, float) for score in scores)
            assert all(0 <= score <= 1 for score in scores)


def test_sync_client_error_handling():
    """Test error handling in synchronous client"""
    with pytest.raises(Exception):
        with ScoringClient("http://invalid-url") as client:
            client.score_batch(TEST_MESSAGES["original"], TEST_MESSAGES["compressed"])


@pytest.mark.asyncio
async def test_async_client_error_handling():
    """Test error handling in asynchronous client"""
    with pytest.raises(Exception):
        async with AsyncScoringClient("http://invalid-url") as client:
            await client.score_batch(
                TEST_MESSAGES["original"], TEST_MESSAGES["compressed"]
            )
