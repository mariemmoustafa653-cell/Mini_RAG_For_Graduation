import sys
from fastapi.testclient import TestClient
from loguru import logger
from app.config import settings
from app.main import app

def run_tests():
    # Make sure we have a test key
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "sk-your-key-here":
        # Provide a dummy key for testing so it doesn't crash on pydantic validation, 
        # but the API calls will fail if it's not a real key.
        # Actually we shouldn't mock it if we want to test end-to-end. We'll rely on the real env.
        pass

    logger.info("Starting Backend Verification...")
    client = TestClient(app)

    # 1. Health Check
    logger.info("1. Testing /health")
    response = client.get("/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    logger.info("   ✅ /health passed")

    # 2. Actions List
    logger.info("2. Testing /api/actions")
    response = client.get("/api/actions")
    assert response.status_code == 200, f"Actions check failed: {response.text}"
    logger.info("   ✅ /api/actions passed")

    # 3. Stats Endpoint
    logger.info("3. Testing /api/stats")
    response = client.get("/api/stats")
    assert response.status_code == 200, f"Stats check failed: {response.text}"
    logger.info("   ✅ /api/stats passed")

    # 4. Chat endpoint (should return hallucination guard / NO_CONTEXT without a real document)
    logger.info("4. Testing /api/chat with dummy data (checking RAG pipeline handling of empty docs)")
    response = client.post("/api/chat", json={
        "teacher_id": "test_verification_user",
        "message": "What is the meaning of life?",
        "doc_id": None
    })
    
    if response.status_code == 500:
        logger.error(f"   ❌ /api/chat threw 500 Internal Server Error: {response.text}")
        sys.exit(1)
        
    assert response.status_code == 200, f"Chat endpoint failed with status {response.status_code}: {response.text}"
    
    # Check if the hallucination guard successfully caught the missing context
    response_data = response.json()
    assert "I don't know" in response_data["response"] or "context" in response_data["response"].lower(), \
        f"AI hallucinated instead of rejecting unknown query: {response_data['response']}"
    
    logger.info("   ✅ /api/chat passed (hallucination guard working properly)")

    logger.info("All Verification Tests Passed Successfully! 🎉")

if __name__ == "__main__":
    run_tests()
