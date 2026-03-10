
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


class OpenAIResponsesJSONClient:
    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model

    def _image_to_data_url(self, image_path: str) -> str:
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type is None:
            mime_type = 'image/png'
        data = base64.b64encode(path.read_bytes()).decode('utf-8')
        return f'data:{mime_type};base64,{data}'

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith('```'):
            text = text.strip('`')
            if text.startswith('json'):
                text = text[4:].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}')
            if start >= 0 and end > start:
                return json.loads(text[start:end+1])
            raise

    def invoke_json(self, prompt: str, image_paths: Optional[List[str]] = None) -> Dict[str, Any]:
        content: List[Dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for image_path in image_paths or []:
            content.append({
                "type": "input_image",
                "image_url": self._image_to_data_url(image_path),
            })

        response = self.client.responses.create(
            model=self.model,
            input=[{"role": "user", "content": content}],
        )
        return self._extract_json(response.output_text)
