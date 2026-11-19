"""Verify prediction and review endpoints on a migrated release."""
import json
import sys
from urllib.error import HTTPError
from urllib.request import Request, urlopen

url = sys.argv[1]


def request(path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = Request(url + path, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=5) as response:
        assert response.status == 200
        return json.load(response)


result = request("/predict", {"id": 123, "full_name": "CI smoke"})
assert result["prediction"] in (0, 1) and 0 <= result["predict_proba"] <= 1
assert request("/applications/123/review") == {"id": 123, "review_status": "pending"}
try:
    request("/applications/-1/review")
except HTTPError as error:
    assert error.code == 404, error
else:
    raise AssertionError("Missing application should return 404")
print("Prediction, review status, and missing-application check: PASS")
