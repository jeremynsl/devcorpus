import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from scraper_chat.plan_mode.plan_mode import PlanModeExecutor


@pytest.fixture
def mock_chroma():
    handler = MagicMock()
    handler.query.return_value = [
        {
            "text": "Sample documentation text",
            "url": "https://example.com/doc1",
            "distance": 0.1,
            "metadata": {"url": "https://example.com/doc1"},
        }
    ]
    return handler


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    # Mock streaming response
    mock_response = [
        MagicMock(choices=[MagicMock(delta=MagicMock(content="1. Design "))]),
        MagicMock(choices=[MagicMock(delta=MagicMock(content="UI"))]),
    ]
    llm.get_response.return_value.__aiter__.return_value = mock_response
    return llm


@pytest.fixture
def plan_mode_executor(mock_chroma, mock_llm):
    with (
        patch(
            "scraper_chat.plan_mode.plan_mode.ChromaHandler", return_value=mock_chroma
        ),
        patch("scraper_chat.plan_mode.plan_mode.LLMConfig", return_value=mock_llm),
    ):
        return PlanModeExecutor(["test_collection"], "test_model")


def test_parse_plan(plan_mode_executor):
    """Test parsing of plan text into steps."""
    # Test with numbered steps
    plan_text = "1. Design the user interface\n2. Implement backend logic\n3. Add authentication"
    parsed_steps = plan_mode_executor.parse_plan(plan_text)
    assert len(parsed_steps) == 3
    assert parsed_steps[0] == "1. Design the user interface"
    assert parsed_steps[1] == "2. Implement backend logic"
    assert parsed_steps[2] == "3. Add authentication"

    # Test with non-numbered steps
    plan_text_alt = "- Design the user interface\n- Implement backend logic"
    parsed_steps_alt = plan_mode_executor.parse_plan(plan_text_alt)
    assert len(parsed_steps_alt) == 2
    assert parsed_steps_alt[0] == "- Design the user interface"
    assert parsed_steps_alt[1] == "- Implement backend logic"


@pytest.mark.asyncio
async def test_generate_plan(plan_mode_executor):
    """Test plan generation."""
    user_message = "Create a web application for task management"

    # Collect plan chunks
    plan_chunks = []
    async for chunk in plan_mode_executor.generate_plan(user_message):
        plan_chunks.append(chunk)

    # Verify chunks
    assert "".join(plan_chunks).strip() == "1. Design UI"


@pytest.mark.asyncio
async def test_generate_docs_query(plan_mode_executor):
    """Test generating a search query for a step."""
    step = "Design the user interface for a task management app"

    # Collect query chunks
    query_chunks = []
    async for chunk in plan_mode_executor.generate_docs_query(step):
        query_chunks.append(chunk)

    # Verify query generation
    assert len(query_chunks) > 0


def test_retrieve_docs(plan_mode_executor):
    """Test document retrieval."""
    query_text = "svelte task management UI"
    docs = plan_mode_executor.retrieve_docs(query_text)

    # Verify retrieval
    assert len(docs) == 1
    assert docs[0]["url"] == "https://example.com/doc1"


@pytest.mark.asyncio
async def test_generate_solution_for_step(plan_mode_executor):
    """Test solution generation for a step."""
    step = "Design the user interface for a task management app"
    docs = [{"text": "Sample doc", "url": "example.com", "distance": 0.1}]
    history = []

    # Collect solution chunks
    solution_chunks = []
    async for chunk in plan_mode_executor.generate_solution_for_step(
        step, docs, history
    ):
        solution_chunks.append(chunk)

    # Verify solution generation
    assert len(solution_chunks) > 0


@pytest.mark.asyncio
async def test_plan_and_execute(plan_mode_executor):
    """Test the entire plan and execute workflow."""
    user_message = "Create a web application for task management"

    # Collect yielded values
    # Collect yielded PlanOutput objects
    yielded_outputs = []
    async for output in plan_mode_executor.plan_and_execute(user_message):
        # Now output is a PlanOutput instance
        yielded_outputs.append(output)

    # Verify workflow
    assert any("Execution Phase" in output.message for output in yielded_outputs)
    assert any(
        "Plan Execution Completed" in output.message for output in yielded_outputs
    )
