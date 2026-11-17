import os
import pytest

@pytest.mark.skipif(True, reason="Enable after server is running in CI")
def test_search_endpoint():
    import requests
    r = requests.get("http://localhost:8080/api/search", params={"q": "dog"})
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
