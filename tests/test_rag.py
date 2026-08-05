"""
Tests for RAG Assistant.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.routes import QueryRequest, SourceResponse
from src.ingestion.chunker import (
    chunk_document,
    clean_html,
    merge_tiny_chunks,
    normalize_text,
    remove_frontmatter,
    split_by_headings,
    split_by_length,
    split_long_section,
)
from src.rag.pipeline import RagPipeline, RagResponse, RetrievedChunk

# =============================================================================
# Chunker Tests
# =============================================================================

class TestChunker:
    """Tests for document chunking."""
    
    def test_clean_html_removes_tags(self):
        html = "<p>Hello <b>world</b></p><script>alert('xss')</script>"
        cleaned = clean_html(html)
        assert "Hello" in cleaned
        assert "world" in cleaned
        assert "<script>" not in cleaned
        assert "alert" not in cleaned
    
    def test_remove_frontmatter(self):
        text = "---\ntitle: Test\n---\n# Content\nBody"
        cleaned = remove_frontmatter(text)
        assert "title:" not in cleaned
        assert "# Content" in cleaned
    
    def test_normalize_text(self):
        text = "Hello\u00a0world\u2014test\u201cquote\u201d"
        normalized = normalize_text(text)
        assert "Hello world" in normalized
        assert "--" in normalized
        assert '"quote"' in normalized
    
    def test_split_by_headings(self):
        md = "# H1\nContent 1\n## H2\nContent 2\n### H3\nContent 3"
        sections = split_by_headings(md)
        assert len(sections) == 3
        assert sections[0]["heading"] == "H1"
        assert sections[1]["heading"] == "H1 > H2"
        assert sections[2]["heading"] == "H1 > H2 > H3"
    
    def test_split_long_section(self):
        long_text = "Para 1\n\nPara 2\n\nPara 3"
        chunks = split_long_section("Test", long_text)
        assert len(chunks) >= 1
    
    def test_split_by_length(self):
        text = "word " * 500
        chunks = split_by_length(text)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 1200
    
    def test_merge_tiny_chunks(self):
        chunks = [
            {"heading": "H1", "text": "Large chunk with enough content here to make it big enough to exceed the minimum chunk size threshold which is two hundred characters so this needs to be quite a long piece of text that will not be considered tiny by the merging algorithm and this makes it definitely over two hundred characters long enough to not be merged"},
            {"heading": "H1", "text": "Tiny"},
            {"heading": "H2", "text": "Another large chunk here with enough content to exceed the minimum chunk size threshold which is two hundred characters so this needs to be quite a long piece of text that is definitely over two hundred characters long"},
]
        merged = merge_tiny_chunks(chunks)
        assert len(merged) == 2
        assert "Tiny" in merged[0]["text"]

    def test_chunk_document_integration(self, tmp_path):
        md_file = tmp_path / "test.md"
        
        intro = "This is the introduction with some content that is long enough to be its own chunk and exceed the minimum size threshold. " + "This is additional content to make the chunk long enough to not be merged. " * 10
        section1 = "Content for section 1 that is sufficiently long enough to be a separate chunk on its own and exceed the minimum size threshold. " + "More content to ensure this section is long enough. " * 10
        subsection = "Content for subsection that is also long enough to be a separate chunk and exceed the minimum size threshold. " + "Even more content for the subsection. " * 10
        section2 = "Content for section 2 that is long enough to be its own chunk and exceed the minimum size threshold. " + "Final section content to make it long enough. " * 10
        
        md_content = f"# Introduction\n{intro}\n\n## Section 1\n{section1}\n\n### Subsection 1.1\n{subsection}\n\n## Section 2\n{section2}"
        md_file.write_text(md_content)
        
        chunks = chunk_document(md_file)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert "heading" in chunk
            assert "text" in chunk
            assert len(chunk["text"]) > 0


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestRagPipeline:
    """Tests for RAG pipeline."""
    
    @pytest.fixture
    def mock_pipeline(self):
        pipeline = RagPipeline()
        pipeline._initialized = True
        pipeline.embed_model = MagicMock()
        pipeline.collection = MagicMock()
        return pipeline
    
    def test_build_prompt(self, mock_pipeline):
        chunks = [
            RetrievedChunk(
                id="1", text="FastAPI is a web framework", 
                heading="Overview", source="index", score=0.9, chars=30
            ),
            RetrievedChunk(
                id="2", text="Use Query for query params", 
                heading="Query Params", source="query", score=0.8, chars=25
            ),
        ]
        prompt = mock_pipeline.build_prompt("What is FastAPI?", chunks)
        assert "FastAPI is a web framework" in prompt
        assert "Use Query for query params" in prompt
        assert "What is FastAPI?" in prompt
        assert "[Overview]" in prompt
        assert "[Query Params]" in prompt
    
    @pytest.mark.asyncio
    async def test_generate_calls_groq(self, mock_pipeline):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "FastAPI is a modern web framework"}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            result = await mock_pipeline.generate("Test prompt")
            assert result == "FastAPI is a modern web framework"
    
    @pytest.mark.asyncio
    async def test_answer_returns_response(self, mock_pipeline):
        mock_pipeline.retrieve = MagicMock(return_value=[
            RetrievedChunk(id="1", text="Content", heading="H1", source="src", score=0.9, chars=10)
        ])
        mock_pipeline.generate = AsyncMock(return_value="Generated answer")
        
        response = await mock_pipeline.answer("Test question")
        
        assert isinstance(response, RagResponse)
        assert response.answer == "Generated answer"
        assert len(response.sources) == 1
        assert response.latency_ms >= 0


# =============================================================================
# API Tests
# =============================================================================

class TestAPIRoutes:
    """Tests for API routes."""
    
    def test_query_request_validation(self):
        req = QueryRequest(question="What is FastAPI?", top_k=4)
        assert req.question == "What is FastAPI?"
        assert req.top_k == 4
        assert req.temperature == 0.1
        
        with pytest.raises(ValueError):
            QueryRequest(question="Test", top_k=0)
        
        with pytest.raises(ValueError):
            QueryRequest(question="Test", top_k=21)
        
        with pytest.raises(ValueError):
            QueryRequest(question="Test", temperature=-0.1)
        
        with pytest.raises(ValueError):
            QueryRequest(question="Test", temperature=2.1)
    
    def test_source_response_model(self):
        source = SourceResponse(
            id="1", text="Content", heading="H1", source="src", score=0.9, chars=10
        )
        assert source.score == 0.9
        assert source.heading == "H1"


# =============================================================================
# Evaluation Tests
# =============================================================================

class TestEvaluation:
    """Tests for evaluation metrics."""
    
    def test_context_precision(self):
        from src.eval.runner import _calc_context_precision
        
        retrieved = ["FastAPI validates query params", "Use Path for path params"]
        expected = ["query params", "validation"]
        
        precision = _calc_context_precision(retrieved, expected)
        assert precision == 0.5
    
    def test_context_recall(self):
        from src.eval.runner import _calc_context_recall
        
        retrieved = ["FastAPI validates query params with Query"]
        expected = ["query params", "validates", "Query"]
        
        recall = _calc_context_recall(retrieved, expected)
        assert recall == 1.0
    
    def test_faithfulness(self):
        from src.eval.runner import _calc_faithfulness
        
        answer = "FastAPI uses Query for query parameter validation"
        context = ["FastAPI validates query params with Query class for validation"]
        
        faithfulness = _calc_faithfulness(answer, context)
        assert faithfulness > 0.5
    
    def test_answer_relevancy(self):
        from src.eval.runner import _calc_answer_relevancy
        
        answer = "FastAPI validates query parameters using the Query class"
        question = "How do I validate query parameters in FastAPI?"
        
        relevancy = _calc_answer_relevancy(answer, question)
        assert relevancy > 0.3


# =============================================================================
# Integration Tests (require services)
# =============================================================================

@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring running services."""
    
    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])