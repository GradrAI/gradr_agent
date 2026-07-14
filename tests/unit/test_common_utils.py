import pytest
import json
import logging
from unittest.mock import patch, MagicMock

from app.app_utils.common import _clean_json, _log_agent_complete

def test_clean_json_valid_json():
    raw = '{"key": "value"}'
    assert _clean_json(raw) == raw

def test_clean_json_with_markdown():
    raw = '```json\n{"key": "value"}\n```'
    assert _clean_json(raw) == '{"key": "value"}'

def test_clean_json_with_surrounding_text():
    raw = 'Here is the JSON you requested: {"key": "value"} Hope this helps!'
    assert _clean_json(raw) == '{"key": "value"}'

def test_clean_json_list():
    raw = '```json\n[{"key": "value"}]\n```'
    assert _clean_json(raw) == '[{"key": "value"}]'

def test_clean_json_list_with_text():
    raw = 'Here are the items: [{"key": "value"}] Done.'
    assert _clean_json(raw) == '[{"key": "value"}]'

def test_clean_json_complex_nesting():
    raw = 'Some text {"outer": {"inner": [1, 2, 3]}} More text'
    assert _clean_json(raw) == '{"outer": {"inner": [1, 2, 3]}}'

@patch('app.app_utils.common.logger')
def test_log_agent_complete(mock_logger):
    _log_agent_complete("TestAgent", "test_output_key")
    
    mock_logger.info.assert_called_once()
    logged_call = mock_logger.info.call_args[0][0]
    
    # Verify the logged string is valid JSON and contains the expected fields
    logged_json = json.loads(logged_call)
    assert logged_json["agent"] == "TestAgent"
    assert logged_json["status"] == "complete"
    assert logged_json["output_key"] == "test_output_key"
    assert "timestamp" in logged_json
