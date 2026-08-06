"""
Evaluation Runner - Ragas-based evaluation for RAG quality.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles

from src.rag.pipeline import RagPipeline
from src.shared import get_logger, track_operation

logger = get_logger(__name__)

DEFAULT_DATASET = [
    {
        "question": "How do I send a GET request with query parameters using httpx?",
        "expected_topics": ["params", "get", "query"],
        "ground_truth": "Use httpx.get(url, params={...}) or pass params= to a Client request."
    },
    {
        "question": "What is the difference between httpx.Client and httpx.AsyncClient?",
        "expected_topics": ["Client", "AsyncClient", "async", "sync"],
        "ground_truth": "Client is for synchronous requests, AsyncClient is for asyncio-based async/await requests."
    },
    {
        "question": "How is authentication implemented in httpx?",
        "expected_topics": ["Auth", "auth_flow", "BasicAuth"],
        "ground_truth": "httpx defines an Auth base class with an auth_flow method; BasicAuth and DigestAuth subclass it."
    },
    {
        "question": "What exceptions can httpx raise on a timeout?",
        "expected_topics": ["Timeout", "TimeoutException", "exceptions"],
        "ground_truth": "httpx raises TimeoutException and its subclasses (ConnectTimeout, ReadTimeout, WriteTimeout, PoolTimeout)."
    },
    {
        "question": "How does httpx handle connection pooling and limits?",
        "expected_topics": ["Limits", "pool", "connections", "keepalive"],
        "ground_truth": "The Limits class configures max_connections, max_keepalive_connections, and keepalive_expiry."
    },
    {
        "question": "How do I use a custom transport in httpx?",
        "expected_topics": ["transport", "Transport", "mounts"],
        "ground_truth": "Pass a transport= argument to Client, or use mounts= to map URL patterns to transports."
    },
    {
        "question": "What does the Client.request method do internally?",
        "expected_topics": ["request", "build_request", "send"],
        "ground_truth": "It builds a Request object via build_request and passes it to send() for the actual network call."
    },
    {
        "question": "How do I stream a response body in httpx?",
        "expected_topics": ["stream", "iter_bytes", "iter_text"],
        "ground_truth": "Use client.stream() as a context manager, then iterate with response.iter_bytes() or iter_text()."
    },
    {
        "question": "What HTTP versions does httpx support?",
        "expected_topics": ["HTTP/2", "HTTP/1.1", "http2"],
        "ground_truth": "httpx supports HTTP/1.1 by default and HTTP/2 when installed with the http2 extra."
    },
    {
        "question": "How does httpx represent a URL internally?",
        "expected_topics": ["URL", "scheme", "host", "path"],
        "ground_truth": "The URL class parses and stores scheme, host, port, path, query, and fragment components."
    },
]


async def run_evaluation_stream(
    pipeline: RagPipeline,
    dataset_name: str = "default",
    metrics: list[str] | None = None,
):
    """Run evaluation on the RAG pipeline, yielding a progress event after
    each question. A full run makes one pipeline call (retrieval +
    generation) per question - worth several seconds each - so a caller
    driving a progress bar needs per-question updates, not just a final
    result after a silent wait."""
    metrics = metrics or ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

    logger.info("evaluation_started", dataset=dataset_name, metrics=metrics)

    dataset = DEFAULT_DATASET
    results = []
    total = len(dataset)

    for i, item in enumerate(dataset, start=1):
        question = item["question"]
        expected_topics = item["expected_topics"]

        yield {"current": i, "total": total, "question": question, "stage": "querying"}
        response = await pipeline.answer(question, top_k=4)

        retrieved_texts = [chunk.text for chunk in response.sources]

        context_precision = _calc_context_precision(retrieved_texts, expected_topics)
        context_recall = _calc_context_recall(retrieved_texts, expected_topics)
        faithfulness = _calc_faithfulness(response.answer, retrieved_texts)
        answer_relevancy = _calc_answer_relevancy(response.answer, question)

        passed = all(m >= 0.5 for m in [context_precision, context_recall, faithfulness, answer_relevancy])

        result = {
            "question": question,
            "answer": response.answer,
            "sources": [{"id": c.id, "heading": c.heading, "score": c.score} for c in response.sources],
            "context_precision": context_precision,
            "context_recall": context_recall,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "latency_ms": response.latency_ms,
            "passed": passed,
        }
        results.append(result)
        yield {"current": i, "total": total, "question": question, "stage": "scored", "passed": passed}

        logger.info(
            "eval_item_completed",
            question=question[:50],
            context_precision=context_precision,
            context_recall=context_recall,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            passed=passed,
        )

    aggregate = {
        "context_precision": sum(r["context_precision"] for r in results) / len(results),
        "context_recall": sum(r["context_recall"] for r in results) / len(results),
        "faithfulness": sum(r["faithfulness"] for r in results) / len(results),
        "answer_relevancy": sum(r["answer_relevancy"] for r in results) / len(results),
        "pass_rate": sum(1 for r in results if r["passed"]) / len(results),
    }

    output_dir = Path("eval/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "dataset": dataset_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregate": aggregate,
        "results": results,
    }

    async with aiofiles.open(output_dir / "latest.json", "w") as f:
        await f.write(json.dumps(output, indent=2))

    async with aiofiles.open(output_dir / f"{dataset_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        await f.write(json.dumps(output, indent=2))

    logger.info("evaluation_completed", aggregate=aggregate)
    yield {"done": True, "output": output}


@track_operation("run_evaluation")
async def run_evaluation(
    pipeline: RagPipeline,
    dataset_name: str = "default",
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Non-streaming convenience wrapper - runs the generator to
    completion and returns the final aggregate output."""
    output = None
    async for event in run_evaluation_stream(pipeline, dataset_name, metrics):
        if event.get("done"):
            output = event["output"]
    return output


def _calc_context_precision(retrieved_texts: list[str], expected_topics: list[str]) -> float:
    if not retrieved_texts:
        return 0.0
    
    relevant_count = 0
    for text in retrieved_texts:
        text_lower = text.lower()
        if any(topic.lower() in text_lower for topic in expected_topics):
            relevant_count += 1
    
    return relevant_count / len(retrieved_texts)


def _calc_context_recall(retrieved_texts: list[str], expected_topics: list[str]) -> float:
    if not expected_topics:
        return 1.0
    
    found_topics = set()
    for text in retrieved_texts:
        text_lower = text.lower()
        for topic in expected_topics:
            if topic.lower() in text_lower:
                found_topics.add(topic)
    
    return len(found_topics) / len(expected_topics)


def _calc_faithfulness(answer: str, retrieved_texts: list[str]) -> float:
    answer_lower = answer.lower()
    
    if "don't know" in answer_lower or "don't have" in answer_lower:
        return 0.8
    
    context_combined = " ".join(retrieved_texts).lower()
    answer_words = set(answer_lower.split())
    context_words = set(context_combined.split())
    
    content_words = {w for w in answer_words if len(w) > 3 and w.isalpha()}
    if not content_words:
        return 0.5
    
    grounded_words = content_words & context_words
    return len(grounded_words) / len(content_words)


def _calc_answer_relevancy(answer: str, question: str) -> float:
    question_lower = question.lower()
    answer_lower = answer.lower()
    
    import re
    question_terms = set(re.findall(r'\b\w{4,}\b', question_lower))
    question_terms = {t for t in question_terms if t not in {'what', 'how', 'does', 'fastapi', 'the', 'and', 'for', 'with', 'you'}}
    
    if not question_terms:
        return 0.5
    
    answer_terms = set(re.findall(r'\b\w{4,}\b', answer_lower))
    overlap = question_terms & answer_terms
    
    return len(overlap) / len(question_terms)


async def run_ragas_evaluation(pipeline: RagPipeline, dataset_name: str = "default") -> dict[str, Any]:
    """Run evaluation using Ragas (requires eval dependencies)."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        logger.warning("ragas_not_installed", message="Install ragas and datasets for full evaluation")
        return await run_evaluation(pipeline, dataset_name)
    
    dataset = DEFAULT_DATASET
    
    questions = []
    answers = []
    contexts = []
    ground_truths = []
    
    for item in dataset:
        response = await pipeline.answer(item["question"], top_k=4)
        questions.append(item["question"])
        answers.append(response.answer)
        contexts.append([chunk.text for chunk in response.sources])
        ground_truths.append(item["ground_truth"])
    
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })
    
    result = evaluate(
        eval_dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
    )
    
    return result.to_pandas().to_dict()