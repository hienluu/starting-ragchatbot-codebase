import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.api
class TestQueryEndpoint:
    """Test suite for /api/query endpoint"""

    def test_query_with_session_id(self, test_client, mock_rag_system):
        """Test query endpoint with provided session ID"""
        response = test_client.post(
            "/api/query",
            json={
                "query": "What is MCP?",
                "session_id": "existing_session_123"
            }
        )

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

        # Verify content
        assert data["answer"] == "This is a test response about MCP."
        assert data["session_id"] == "existing_session_123"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Test Course - Lesson 1"
        assert data["sources"][0]["link"] == "https://example.com/lesson1"

        # Verify RAG system was called correctly
        mock_rag_system.query.assert_called_once_with("What is MCP?", "existing_session_123")

    def test_query_without_session_id_creates_new_session(self, test_client, mock_rag_system):
        """Test query endpoint creates new session when not provided"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify new session was created
        assert data["session_id"] == "test_session_123"
        mock_rag_system.session_manager.create_session.assert_called_once()

        # Verify query was executed with new session
        mock_rag_system.query.assert_called_once_with("What is MCP?", "test_session_123")

    def test_query_with_empty_query_string(self, test_client, mock_rag_system):
        """Test query endpoint with empty query string"""
        # Configure mock to handle empty query
        mock_rag_system.query.return_value = (
            "Please provide a question.",
            []
        )

        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data

    def test_query_missing_required_field(self, test_client):
        """Test query endpoint with missing required 'query' field"""
        response = test_client.post(
            "/api/query",
            json={"session_id": "test"}
        )

        assert response.status_code == 422  # Unprocessable Entity (validation error)

    def test_query_invalid_json(self, test_client):
        """Test query endpoint with invalid JSON"""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_query_with_sources_empty(self, test_client, mock_rag_system):
        """Test query endpoint when no sources are returned"""
        mock_rag_system.query.return_value = (
            "This is general knowledge, no course content needed.",
            []
        )

        response = test_client.post(
            "/api/query",
            json={"query": "What is machine learning?"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == []

    def test_query_error_handling(self, test_client, mock_rag_system):
        """Test query endpoint handles RAG system errors gracefully"""
        mock_rag_system.query.side_effect = Exception("Database connection error")

        response = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )

        assert response.status_code == 500
        assert "Database connection error" in response.json()["detail"]

    def test_query_with_special_characters(self, test_client, mock_rag_system):
        """Test query endpoint with special characters in query"""
        mock_rag_system.query.return_value = ("Response", [])

        response = test_client.post(
            "/api/query",
            json={"query": "What is <MCP> & how does it work? 🚀"}
        )

        assert response.status_code == 200
        mock_rag_system.query.assert_called_once()

    def test_query_with_very_long_query(self, test_client, mock_rag_system):
        """Test query endpoint with very long query string"""
        long_query = "What is MCP? " * 100  # Very long query
        mock_rag_system.query.return_value = ("Response", [])

        response = test_client.post(
            "/api/query",
            json={"query": long_query}
        )

        assert response.status_code == 200

    def test_query_response_model_validation(self, test_client, mock_rag_system):
        """Test that response adheres to QueryResponse model"""
        response = test_client.post(
            "/api/query",
            json={"query": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields exist
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)

        # Verify source structure
        if len(data["sources"]) > 0:
            source = data["sources"][0]
            assert "text" in source
            assert "link" in source


@pytest.mark.api
class TestCoursesEndpoint:
    """Test suite for /api/courses endpoint"""

    def test_get_course_stats_success(self, test_client, mock_rag_system):
        """Test successful retrieval of course statistics"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Verify structure
        assert "total_courses" in data
        assert "course_titles" in data

        # Verify content
        assert data["total_courses"] == 2
        assert len(data["course_titles"]) == 2
        assert "Introduction to MCP" in data["course_titles"]
        assert "Advanced MCP" in data["course_titles"]

        # Verify RAG system was called
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_course_stats_empty_courses(self, test_client, mock_rag_system):
        """Test course stats when no courses are loaded"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_course_stats_error_handling(self, test_client, mock_rag_system):
        """Test course stats endpoint handles errors gracefully"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Database error")

        response = test_client.get("/api/courses")

        assert response.status_code == 500
        assert "Database error" in response.json()["detail"]

    def test_get_course_stats_large_number_of_courses(self, test_client, mock_rag_system):
        """Test course stats with large number of courses"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 100,
            "course_titles": [f"Course {i}" for i in range(100)]
        }

        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 100
        assert len(data["course_titles"]) == 100

    def test_get_course_stats_response_model_validation(self, test_client):
        """Test that response adheres to CourseStats model"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Verify types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)
        assert all(isinstance(title, str) for title in data["course_titles"])


@pytest.mark.api
class TestSessionEndpoint:
    """Test suite for /api/session/{session_id} endpoint"""

    def test_delete_session_success(self, test_client, mock_rag_system):
        """Test successful session deletion"""
        response = test_client.delete("/api/session/test_session_123")

        assert response.status_code == 200
        data = response.json()

        # Verify response
        assert data["success"] is True
        assert data["message"] == "Session deleted"

        # Verify session manager was called
        mock_rag_system.session_manager.delete_session.assert_called_once_with("test_session_123")

    def test_delete_nonexistent_session(self, test_client, mock_rag_system):
        """Test deleting a session that doesn't exist"""
        mock_rag_system.session_manager.delete_session.return_value = False

        response = test_client.delete("/api/session/nonexistent_session")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is False
        assert data["message"] == "Session not found"

    def test_delete_session_error_handling(self, test_client, mock_rag_system):
        """Test session deletion error handling"""
        mock_rag_system.session_manager.delete_session.side_effect = Exception("Session error")

        response = test_client.delete("/api/session/test_session_123")

        assert response.status_code == 500
        assert "Session error" in response.json()["detail"]

    def test_delete_session_with_special_characters(self, test_client, mock_rag_system):
        """Test deleting session with special characters in ID"""
        session_id = "session-123_test@2024"
        mock_rag_system.session_manager.delete_session.return_value = True

        response = test_client.delete(f"/api/session/{session_id}")

        assert response.status_code == 200
        mock_rag_system.session_manager.delete_session.assert_called_once_with(session_id)

    def test_delete_session_with_very_long_id(self, test_client, mock_rag_system):
        """Test deleting session with very long ID"""
        long_session_id = "x" * 500
        mock_rag_system.session_manager.delete_session.return_value = True

        response = test_client.delete(f"/api/session/{long_session_id}")

        assert response.status_code == 200


@pytest.mark.api
class TestHealthEndpoint:
    """Test suite for health check endpoint"""

    def test_health_check(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"


@pytest.mark.api
class TestCORSAndMiddleware:
    """Test suite for CORS and middleware configuration"""

    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are properly set"""
        response = test_client.options(
            "/api/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )

        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers

    def test_cors_allows_all_origins(self, test_client, mock_rag_system):
        """Test that CORS allows requests from any origin"""
        response = test_client.post(
            "/api/query",
            json={"query": "test"},
            headers={"Origin": "http://example.com"}
        )

        assert response.status_code == 200
        assert response.headers.get("access-control-allow-origin") == "*"


@pytest.mark.api
class TestEndToEndAPIFlows:
    """Integration tests for complete API workflows"""

    def test_complete_conversation_flow(self, test_client, mock_rag_system):
        """Test complete conversation: create session → query → query → delete"""
        # First query without session (creates new one)
        response1 = test_client.post(
            "/api/query",
            json={"query": "What is MCP?"}
        )
        assert response1.status_code == 200
        session_id = response1.json()["session_id"]

        # Second query with same session
        mock_rag_system.query.return_value = ("Follow-up answer", [])
        response2 = test_client.post(
            "/api/query",
            json={"query": "Tell me more", "session_id": session_id}
        )
        assert response2.status_code == 200
        assert response2.json()["session_id"] == session_id

        # Delete session
        response3 = test_client.delete(f"/api/session/{session_id}")
        assert response3.status_code == 200
        assert response3.json()["success"] is True

    def test_query_and_get_courses_flow(self, test_client, mock_rag_system):
        """Test querying and then checking available courses"""
        # Query first
        response1 = test_client.post(
            "/api/query",
            json={"query": "What courses are available?"}
        )
        assert response1.status_code == 200

        # Get course stats
        response2 = test_client.get("/api/courses")
        assert response2.status_code == 200
        assert response2.json()["total_courses"] >= 0

    def test_multiple_concurrent_sessions(self, test_client, mock_rag_system):
        """Test handling multiple concurrent sessions"""
        mock_rag_system.session_manager.create_session.side_effect = [
            "session_1",
            "session_2",
            "session_3"
        ]

        # Create three sessions
        responses = []
        for i in range(3):
            response = test_client.post(
                "/api/query",
                json={"query": f"Query {i}"}
            )
            responses.append(response)

        # Verify all succeeded and have different session IDs
        assert all(r.status_code == 200 for r in responses)
        session_ids = [r.json()["session_id"] for r in responses]
        assert len(set(session_ids)) == 3  # All unique


@pytest.mark.api
class TestErrorScenarios:
    """Test various error scenarios and edge cases"""

    def test_invalid_http_method_on_query(self, test_client):
        """Test using wrong HTTP method on query endpoint"""
        response = test_client.get("/api/query")
        assert response.status_code == 405  # Method Not Allowed

    def test_invalid_http_method_on_courses(self, test_client):
        """Test using wrong HTTP method on courses endpoint"""
        response = test_client.post("/api/courses", json={})
        assert response.status_code == 405  # Method Not Allowed

    def test_nonexistent_endpoint(self, test_client):
        """Test requesting non-existent endpoint"""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_malformed_json_in_body(self, test_client):
        """Test request with malformed JSON body"""
        response = test_client.post(
            "/api/query",
            data="{query: 'missing quotes'}",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_extra_fields_in_request_ignored(self, test_client, mock_rag_system):
        """Test that extra fields in request are ignored"""
        response = test_client.post(
            "/api/query",
            json={
                "query": "test",
                "extra_field": "should be ignored",
                "another_field": 123
            }
        )
        assert response.status_code == 200

    def test_null_values_in_request(self, test_client, mock_rag_system):
        """Test handling of null values in request"""
        response = test_client.post(
            "/api/query",
            json={
                "query": "test",
                "session_id": None
            }
        )
        # Should create new session since session_id is None
        assert response.status_code == 200

    def test_unicode_characters_in_query(self, test_client, mock_rag_system):
        """Test handling of unicode characters"""
        mock_rag_system.query.return_value = ("Response with unicode: 你好", [])

        response = test_client.post(
            "/api/query",
            json={"query": "What about unicode? 你好 мир"}
        )

        assert response.status_code == 200
        assert "你好" in response.json()["answer"]
