"""Tests for the llm_parser module.

Covers basic unit tests for each function in src/llm_parser.py.
"""

from unittest.mock import MagicMock, patch

import pytest

from src import llm_parser


class TestLLMParser:
    def test_create_gemini_client_success(self, monkeypatch):
        monkeypatch.setattr(llm_parser.config, "GEMINI_API_KEY", "dummy-key")
        mock_model = MagicMock()
        mock_instructor = MagicMock()
        monkeypatch.setattr(llm_parser.genai, "GenerativeModel",
                            lambda model_name: mock_model)
        monkeypatch.setattr(llm_parser.instructor, "from_gemini",
                            lambda client, mode: mock_instructor)
        client = llm_parser.create_gemini_client(model="test-model")
        assert client == mock_instructor

    def test_create_gemini_client_no_api_key(self, monkeypatch):
        monkeypatch.setattr(llm_parser.config, "GEMINI_API_KEY", None)
        with pytest.raises(ValueError):
            llm_parser.create_gemini_client()

    def test_create_gemini_client_invalid_model(self, monkeypatch):
        monkeypatch.setattr(llm_parser.config, "GEMINI_API_KEY", "dummy-key")
        with pytest.raises(ValueError):
            llm_parser.create_gemini_client(model=123)

    def test_create_menu_extraction_prompt(self):
        ocr_text = "Sample OCR text"
        prompt = llm_parser.create_menu_extraction_prompt(ocr_text)
        assert ocr_text in prompt
        assert isinstance(prompt, str)

    def test_parse_menu_with_gemini_invalid_input(self, monkeypatch):
        monkeypatch.setattr(llm_parser.config, "GEMINI_API_KEY", "dummy-key")
        with pytest.raises(ValueError):
            llm_parser.parse_menu_with_gemini(123, MagicMock())
        with pytest.raises(ValueError):
            llm_parser.parse_menu_with_gemini("text", "not_a_client")

    def test_parse_menu_with_gemini_empty_ocr(self, monkeypatch):
        monkeypatch.setattr(llm_parser.config, "GEMINI_API_KEY", "dummy-key")
        result = llm_parser.parse_menu_with_gemini("", MagicMock())
        assert result == []

    def test_parse_menu_with_gemini_success(self, monkeypatch):
        monkeypatch.setattr(llm_parser.config, "GEMINI_API_KEY", "dummy-key")


    def test_extract_menu_from_ocr_invalid(self):
        with pytest.raises(ValueError):
            llm_parser.extract_menu_from_ocr(123)
        assert llm_parser.extract_menu_from_ocr("") == []

    def test_extract_menu_from_ocr_success(self, monkeypatch):
        monkeypatch.setattr(
            llm_parser, "create_gemini_client", lambda: "client")
        monkeypatch.setattr(
            llm_parser, "parse_menu_with_gemini", lambda ocr, client: ["menu"])
        result = llm_parser.extract_menu_from_ocr("ocr text")
        assert result == ["menu"]
