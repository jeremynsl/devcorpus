from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, AsyncGenerator
from scraper_chat.database.chroma_handler import ChromaHandler
from scraper_chat.core.llm_config import LLMConfig

@dataclass
class PlanOutput:
    """
    A standardized output message for plan and execution steps.
    
    Attributes:
        message: The content of the message.
        metadata: Optional dictionary containing extra context,
                  such as the phase of the process or the step number.
    """
    message: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlanModeExecutor:
    """
    Handles the two-phase Plan Mode:
    1) Planning Phase: LLM creates a high-level plan (without docs).
    2) Execution Phase: For each step, retrieve docs and generate the solution iteratively.
    """

    def __init__(self, collections, model):
        

        self.db = ChromaHandler()
        self.collections = collections
        self.llm = LLMConfig(model)

    async def generate_plan(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        Ask the LLM for a high-level plan, using RAG context from relevant docs.
        Yields raw plan text chunks that form a JSON array.
        """
        query_prompt = f"""Generate 1 concise search query to find relevant documentation for this request:

Request: "{user_message}"

Guidelines:
1. Include specific technical terms from the request
2. Focus on key concepts and technologies mentioned
3. Keep it concise but specific

Your query:"""

        query = ""
        query_stream = await self.llm.get_response(query_prompt, stream=True)
        async for chunk in query_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                query += chunk.choices[0].delta.content

        docs = self.retrieve_docs(query.strip())
        docs_context = "\n\n".join([doc["text"] for doc in docs])

        plan_prompt = f"""You are in "Plan Mode." The user request is:

{user_message}

Here is relevant documentation context to help form the plan:
{docs_context}

Based on this context and the request, please produce a short, high-level plan with numbered steps.
IMPORTANT: produce the plan in valid JSON with the format:
[
  {{"step_number": 1, "description": "..."}},
  {{"step_number": 2, "description": "..."}},
  ...
]

DO NOT include any extra fields or text outside the JSON.
"""
        self._logger_info("Generating high-level plan with RAG context...")
        response_stream = await self.llm.get_response(plan_prompt, stream=True)
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content  # Stream individual tokens

    def parse_plan(self, plan_text: str) -> List[str]:
        """
        Parse the JSON plan text into a list of step descriptions.
        """
        import json

        MIN_STEPS = 1
        MAX_STEPS = 10

        try:
            plan_data = json.loads(plan_text)
            if not isinstance(plan_data, list):
                self._logger_error("Plan data is not a list")
                raise ValueError("Invalid plan format: expected a list of steps")
            if len(plan_data) < MIN_STEPS:
                self._logger_error("Plan has no steps")
                raise ValueError("Plan must have at least one step")
            if len(plan_data) > MAX_STEPS:
                self._logger_warning(
                    f"Plan has too many steps ({len(plan_data)}), truncating to {MAX_STEPS}"
                )
                plan_data = plan_data[:MAX_STEPS]

            steps = []
            for step in plan_data:
                if not isinstance(step, dict) or "description" not in step:
                    self._logger_error(f"Invalid step format: {step}")
                    continue
                if not step["description"].strip():
                    self._logger_error("Empty step description")
                    continue
                steps.append(step["description"])

            if not steps:
                raise ValueError("No valid steps found in plan")
            return steps

        except json.JSONDecodeError:
            self._logger_error("Failed to parse plan JSON, falling back to text parsing")
            steps = []
            for line in plan_text.split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    steps.append(line)
            if not steps:
                raise ValueError("No valid steps found in plan text")
            if len(steps) > MAX_STEPS:
                self._logger_warning(
                    f"Plan has too many steps ({len(steps)}), truncating to {MAX_STEPS}"
                )
                steps = steps[:MAX_STEPS]
            return steps

    async def generate_docs_query(self, step: str) -> AsyncGenerator[str, None]:
        """
        For each step, produce a short doc-search query.
        """
        prompt = f"""Generate 1 concise search query for implementing this Svelte/SvelteKit step:

Step: "{step}"

Guidelines:
1. Include specific technical terms from the step
2. Add Svelte/SvelteKit framework references
3. Consider relevant libraries (e.g., svelte-calendar, date-fns)
4. Use OR between alternative phrasings
5. Prioritize code examples and official docs

Your queries:
"""
        response_stream = await self.llm.get_response(prompt, stream=True)
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def retrieve_docs(self, query_text: str, n_results: int = 5) -> List[dict]:
        """
        Retrieve documents using the Chroma-based RAG mechanism.
        """
        all_results = []
        for coll in self.collections:
            try:
                results = self.db.query(coll, query_text, n_results=n_results)
                all_results.extend(results)
            except Exception as e:
                self._logger_error(f"Error querying collection {coll}: {str(e)}")
        all_results.sort(key=lambda x: x["distance"])
        top_results = all_results[:n_results]
        self._logger_debug("Final documents selected:")
        for i, doc in enumerate(top_results, 1):
            url = doc.get("metadata", {}).get("url", "No URL available")
            relevance = 1 - doc["distance"]
            self._logger_debug(f"  {i}. {url} (Relevance: {relevance:.2f})")
        return top_results

    async def generate_solution_for_step(self, step: str, docs: List[dict], history: List[dict]) -> AsyncGenerator[str, None]:
        """
        Given a plan step and relevant docs, generate the solution output.
        """
        previous_steps = "\n".join(
            f"Step {s['step_number']}: {s['description']}\n"
            f"Used Query: {s['query']}\n"
            f"Solution: {s['solution']}\n"
            for s in history[:-1]
        )
        doc_context_parts = []
        for i, doc in enumerate(docs, start=1):
            snippet = doc["text"]
            if len(snippet) > 1000:
                snippet = snippet[:1000] + "..."
            doc_context_parts.append(
                f"[{i}] from {doc.get('metadata', {}).get('url', 'No URL available')}\nRelevance: {1 - doc['distance']:.2f}\n\n{snippet}"
            )
        doc_context = (
            "\n\n---\n\n".join(doc_context_parts)
            if doc_context_parts
            else "No docs found."
        )

        prompt = f"""You have the following plan step:

STEP: {step}

Build on previous work:
{previous_steps}

And here are some relevant documentation excerpts:

{doc_context}

Using the above info, produce code or text that addresses this step:
"""
        response_stream = await self.llm.get_response(prompt, stream=True)
        async for chunk in response_stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def plan_and_execute(self, user_message: str, previous_steps: Optional[List[dict]] = None) -> AsyncGenerator[PlanOutput, None]:
        """
        Orchestrate Plan Mode with proper history tracking.
        Yields PlanOutput objects for each piece of output.
        """
        step_history = previous_steps if previous_steps else []

        # Phase 1: Plan Generation
        plan_chunks = []
        async for chunk in self.generate_plan(user_message):
            plan_chunks.append(chunk)
            yield PlanOutput(
                message=chunk,
                metadata={"phase": "plan_generation"}
            )

        parsed_plan = self.parse_plan("".join(plan_chunks))
        yield PlanOutput(
            message="\n\n **Execution Phase**\n",
            metadata={"phase": "phase_transition"}
        )

        # Phase 2: Step Execution
        for step_number, step in enumerate(parsed_plan, start=1):
            current_step = {
                "step_number": step_number,
                "description": step.strip(),
                "query": None,
                "solution": None,
            }
            step_history.append(current_step)
            # Yield step header
            yield PlanOutput(
                message=f"\n\n---\n**Step {step_number}**: {step}\n",
                metadata={"phase": "step_header", "step": step_number}
            )

            # Query Generation
            yield PlanOutput(
                message="🔍 *Searching docs...*\n",
                metadata={"phase": "query_generation", "step": step_number}
            )
            query_chunks = []
            async for chunk in self.generate_docs_query(step):
                query_chunks.append(chunk)
                yield PlanOutput(
                    message=chunk,
                    metadata={"phase": "query_generation", "step": step_number}
                )
            current_step["query"] = "".join(query_chunks).strip()
            relevant_docs = self.retrieve_docs(current_step["query"])

            # Solution Generation
            yield PlanOutput(
                message="\n💡 *Generating solution...*\n",
                metadata={"phase": "solution_generation", "step": step_number}
            )
            solution_chunks = []
            async for chunk in self.generate_solution_for_step(step, relevant_docs, step_history):
                solution_chunks.append(chunk)
                yield PlanOutput(
                    message=chunk,
                    metadata={"phase": "solution_generation", "step": step_number}
                )
            current_step["solution"] = "".join(solution_chunks).strip()

        yield PlanOutput(
            message="\n\n✅ **Plan Execution Completed**",
            metadata={"phase": "completion"}
        )

    # Internal logging helpers for consistency
    def _logger_info(self, msg: str):
        import logging
        logging.getLogger(__name__).info(msg)

    def _logger_error(self, msg: str):
        import logging
        logging.getLogger(__name__).error(msg)

    def _logger_warning(self, msg: str):
        import logging
        logging.getLogger(__name__).warning(msg)

    def _logger_debug(self, msg: str):
        import logging
        logging.getLogger(__name__).debug(msg)
