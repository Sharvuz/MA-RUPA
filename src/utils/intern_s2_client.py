"""
InternLM Discovery (intern-s2) Client for MA-RUPA
Supports OpenAI-compatible API for Shanghai AI Lab's intern-s2 reasoning model.
Features real-time streaming, System 2 reasoning trace capture, and robust error handling.
"""

import os
import json
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Callable


class InternS2Client:
    """Client for InternLM Discovery API (intern-s2 model with System 2 reasoning)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://discovery-api.intern-ai.org.cn/v1",
        model: str = "intern-s2",
        timeout: int = 180
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
        max_tokens: Optional[int] = 2048,
        stream: bool = True,
        on_reasoning_chunk: Optional[Callable[[str], None]] = None,
        on_content_chunk: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        """Send chat completion request to intern-s2 model.
        
        Args:
            messages: List of message dicts with 'role' and 'content'.
            thinking_mode: Enable System 2 deep reasoning mode for intern-s2.
            temperature: Sampling temperature.
            top_p: Nucleus sampling parameter.
            max_tokens: Maximum tokens to generate (defaults to 2048 to allow full reasoning trace).
            stream: Use streaming to prevent HTTP read timeouts and display tokens live.
            on_reasoning_chunk: Optional callback for streaming reasoning tokens.
            on_content_chunk: Optional callback for streaming final content tokens.
        
        Returns:
            Dict containing 'success', 'reasoning', 'content', and raw details.
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
            "stream": stream,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        req = urllib.request.Request(
            endpoint,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) MA-RUPA/2.0"
            },
            data=json.dumps(payload).encode("utf-8")
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if stream:
                    reasoning_chunks: List[str] = []
                    content_chunks: List[str] = []
                    
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0].get("delta", {})
                            
                            # Capture reasoning trace
                            r_text = delta.get("reasoning_content") or delta.get("reasoning")
                            if r_text:
                                reasoning_chunks.append(r_text)
                                if on_reasoning_chunk:
                                    on_reasoning_chunk(r_text)
                            
                            # Capture main answer
                            c_text = delta.get("content")
                            if c_text:
                                content_chunks.append(c_text)
                                if on_content_chunk:
                                    on_content_chunk(c_text)
                        except Exception:
                            continue

                    full_reasoning = "".join(reasoning_chunks).strip()
                    full_content = "".join(content_chunks).strip()

                    return {
                        "success": True,
                        "reasoning": full_reasoning,
                        "content": full_content or full_reasoning,
                    }
                else:
                    data = json.loads(resp.read().decode("utf-8"))
                    msg = data["choices"][0]["message"]
                    return {
                        "success": True,
                        "reasoning": msg.get("reasoning_content", ""),
                        "content": msg.get("content") or msg.get("reasoning_content", ""),
                        "raw": data
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
            f"Format your response strictly as a JSON array of strings."
        )
        res = self.chat_completion(
            messages=[
                {"role": "system", "content": "You are a precise computer vision research assistant. Output strictly JSON."},
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        if res.get("success"):
            content = res.get("content", "")
            try:
                start = content.find("[")
                end = content.rfind("]") + 1
                if start != -1 and end != 0:
                    return json.loads(content[start:end])
            except Exception:
                pass
            return [line.strip("- *0123456789.\" ") for line in content.splitlines() if line.strip()]
        return []
