from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder

from telemetry_anomaly.api import health, segment, summary


def test_read_only_api_exposes_summary_and_test_segment():
    assert health() == {"status": "ok"}
    payload = summary()
    assert payload["selection_basis"] == "internal validation average precision"
    assert payload["test_protocol"]["test_used_for_model_or_threshold_selection"] is False

    record = segment(1276)
    assert record["segment"] == 1276
    json.dumps(jsonable_encoder(record))


def test_api_rejects_segment_not_in_locked_test_split():
    with pytest.raises(HTTPException) as error:
        segment(1)
    assert error.value.status_code == 404
