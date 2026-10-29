"""
InternLM Discovery (intern-s2) Client for MA-RUPA
Supports OpenAI-compatible API for Shanghai AI Lab's intern-s2 reasoning model.
"""

import os
import json
import urllib.request
import urllib.error
from datetime import datetime
from typing import Dict, Any, List, Optional


class InternS2Client:
    """Client for InternLM Discovery API (intern-s2 model with System 2 reasoning)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://discovery-api.intern-ai.org.cn/v1",
        model: str = "intern-s2",
        timeout: int = 60
    ):
        self.api_key = api_key or os.environ.get("INTERN_AI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        thinking_mode: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """Send chat completion request to intern-s2 model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            thinking_mode: Enable System 2 deep reasoning mode for intern-s2.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Maximum tokens to generate.
        
        Returns:
            Dict containing response or error details.
        """
        if not self.api_key:
            raise ValueError(
                "API key is missing! Set INTERN_AI_API_KEY environment variable or pass api_key."
            )

        endpoint = f"{self.base_url}/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "extra_body": {
                "thinking_mode": thinking_mode
            }
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        req = urllib.request.Request(
            endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            data=json.dumps(payload).encode("utf-8")
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choice = data["choices"][0]
                msg = choice.get("message", {})
                content = msg.get("content") or ""
                reasoning = msg.get("reasoning_content") or ""
                
                # If content is empty/None but reasoning exists, keep reasoning accessible
                full_text = content if content else reasoning
                if reasoning and content:
                    combined = f"<thinking>\n{reasoning}\n</thinking>\n\n{content}"
                else:
                    combined = full_text

                return {
                    "success": True,
                    "content": combined,
                    "answer": content,
                    "reasoning": reasoning,
                    "raw": data,
                    "timestamp": datetime.now().isoformat()
                }
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="ignore")
            try:
                err_json = json.loads(err_body)
            except Exception:
                err_json = {"raw_error": err_body}
            return {
                "success": False,
                "status_code": e.code,
                "error": err_json
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def generate_anomaly_descriptors(self, class_name: str, dataset: str = "UCF-Crime") -> List[str]:
        """Use intern-s2 reasoning model to generate fine-grained semantic descriptors for CLIP prompts."""
        prompt = (
            f"You are an expert in video anomaly detection and computer vision. "
            f"For the anomaly category '{class_name}' in dataset '{dataset}', "
            f"provide 5 distinctive, concise visual descriptions of scenes and actions "
            f"characteristic of this anomaly. Each description should be a single clear sentence. "
            f"Format your response as a JSON array of strings."
        )
        res = self.chat_completion(
            messages=[
                {"role": "system", "content": "You are a precise computer vision research assistant."},
                {"role": "user", "content": prompt}
            ],
            thinking_mode=True
        )
        if res.get("success"):
            content = res["content"]
            # Extract JSON list if present
            try:
                start = content.find("[")
                end = content.rfind("]") + 1
                if start != -1 and end != 0:
                    return json.loads(content[start:end])
            except Exception:
                pass
            return [line.strip("- *0123456789. ") for line in content.splitlines() if line.strip()]
        return []
