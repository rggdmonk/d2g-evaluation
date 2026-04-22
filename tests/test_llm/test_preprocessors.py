import json

from llm.preprocessors import FaultTolerantJsonPreprocessor


class TestFaultTolerantJsonPreprocessor:
    def test_strips_code_fences(self):
        preprocessor = FaultTolerantJsonPreprocessor()

        inputs = [
            '```jsonl\n{"annotations": [{"text": "Tips"}]}```',
            '```python\n{"annotations": [{"text": "Tips"}]}```'
            '```javascript\n{"annotations": [{"text": "Tips"}]}```'
            '```typescript\n{"annotations": [{"text": "Tips"}]}```'
            '```json\n{"annotations": [{"text": "Tips"}]}```'
            '```bash\n{"annotations": [{"text": "Tips"}]}```'
            '```markdown\n{"annotations": [{"text": "Tips"}]}```',
        ]
        expected = '{"annotations": [{"text": "Tips"}]}'
        for i in inputs:
            assert preprocessor.process(i) == expected

    def test_fixes_malformed_json(self):
        preprocessor = FaultTolerantJsonPreprocessor()

        json_string = '{"annotations": [{"text": "He said: "Hello!"}]}'
        preprocessed = preprocessor.process(json_string)
        parsed = json.loads(preprocessed)

        assert isinstance(parsed, dict)
        assert parsed["annotations"][0]["text"].startswith('He said: "Hello!"')

    def test_removes_control_chars(self):
        preprocessor = FaultTolerantJsonPreprocessor()

        json_string = '{"annotations": [{"text": "Hello \u00a0\u200b\xa0 world}]}'
        preprocessed = preprocessor.process(json_string)
        parsed = json.loads(preprocessed)

        assert isinstance(parsed, dict)
        # Whitespace handling may change and it's fine
        # as long as there are no control chars here.
        assert all(ord(c) < 128 for c in parsed["annotations"][0]["text"])
        assert parsed["annotations"][0]["text"].startswith("Hello")

    def test_handles_empty_input(self):
        preprocessor = FaultTolerantJsonPreprocessor()
        assert preprocessor.process("") == ""
