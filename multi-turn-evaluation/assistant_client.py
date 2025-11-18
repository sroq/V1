"""
RAG Assistant Backend Kliens - End-to-End kommunikáció.

Ez a modul felelős a Next.js backend API-val való kommunikációért.
POST kérést küld a /api/chat endpointra és feldolgozza a streaming választ.

Backend endpoint:
- URL: http://localhost:3000/api/chat
- Method: POST
- Input: { messages: [{ role: "user"|"assistant", content: string }] }
- Output: Streaming text response (Server-Sent Events)

A kliens támogatja:
- Streaming válaszok kezelését
- Session management (automatikus a backend oldalon)
- Újrapróbálkozás hibák esetén
- Timeout kezelés
"""

import requests
import json
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import time

load_dotenv()


class AssistantClient:
    """
    RAG Assistant Backend kliens.

    Ez a kliens end-to-end kommunikációt biztosít a Next.js backend API-val.
    """

    def __init__(
        self,
        base_url: str = None,
        timeout: int = 60,
        max_retries: int = 3
    ):
        """
        Inicializálás.

        Args:
            base_url: Backend URL (alapértelmezett: http://localhost:3000)
            timeout: Timeout másodpercekben (alapértelmezett: 60)
            max_retries: Maximum újrapróbálkozások száma (alapértelmezett: 3)
        """
        self.base_url = base_url or os.getenv("ASSISTANT_BASE_URL", "http://localhost:3000")
        self.chat_endpoint = f"{self.base_url}/api/chat"
        self.timeout = timeout
        self.max_retries = max_retries

    def send_message(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True
    ) -> Dict[str, Any]:
        """
        Üzenet küldése a backend-nek.

        Args:
            messages: Üzenetek listája [{ role: "user"|"assistant", content: str }]
            stream: Streaming engedélyezése (alapértelmezett: True)

        Returns:
            Dict az eredménnyel:
            {
                "success": bool,
                "response": str,  # Assistant válasza
                "error": Optional[str],
                "metadata": {
                    "duration_ms": int,
                    "response_length": int,
                    "streaming": bool
                },
                "performance": {
                    "total_duration_ms": int,
                    "api_latency_ms": int,
                    "ttft_ms": int,  # Time To First Token (streaming only)
                    "response_tokens": int,  # Approximate token count
                    "tokens_per_second": float,
                    "retry_count": int
                }
            }

        Példa:
            >>> client = AssistantClient()
            >>> result = client.send_message([
            ...     {"role": "user", "content": "Ki az a Maugli?"}
            ... ])
            >>> print(result["response"])
        """
        start_time = time.time()

        # Request payload
        payload = {
            "messages": messages
        }

        # Újrapróbálkozás logika
        for attempt in range(self.max_retries):
            try:
                api_start = time.time()

                if stream:
                    # Streaming request with performance tracking
                    response, perf_metrics = self._send_streaming_request(payload)
                else:
                    # Non-streaming request (egyszerűbb teszteléshez)
                    response = self._send_regular_request(payload)
                    perf_metrics = {}

                api_duration = time.time() - api_start
                total_duration = time.time() - start_time

                # Token counting (approximate: split by whitespace)
                response_tokens = len(response.split())
                tokens_per_sec = response_tokens / api_duration if api_duration > 0 else 0

                return {
                    "success": True,
                    "response": response,
                    "error": None,
                    "metadata": {
                        "duration_ms": int(total_duration * 1000),
                        "response_length": len(response),
                        "streaming": stream,
                        "attempt": attempt + 1
                    },
                    "performance": {
                        "total_duration_ms": int(total_duration * 1000),
                        "api_latency_ms": int(api_duration * 1000),
                        "ttft_ms": perf_metrics.get("ttft_ms", None),
                        "response_tokens": response_tokens,
                        "tokens_per_second": round(tokens_per_sec, 2),
                        "retry_count": attempt
                    }
                }

            except Exception as e:
                error_msg = str(e)
                print(f"Attempt {attempt + 1}/{self.max_retries} failed: {error_msg}")

                if attempt == self.max_retries - 1:
                    # Utolsó próbálkozás, return error
                    duration_ms = int((time.time() - start_time) * 1000)
                    return {
                        "success": False,
                        "response": "",
                        "error": error_msg,
                        "metadata": {
                            "duration_ms": duration_ms,
                            "response_length": 0,
                            "streaming": stream,
                            "attempt": attempt + 1
                        },
                        "performance": {
                            "total_duration_ms": duration_ms,
                            "api_latency_ms": None,
                            "ttft_ms": None,
                            "response_tokens": 0,
                            "tokens_per_second": 0,
                            "retry_count": attempt
                        }
                    }

                # Várakozás újrapróbálkozás előtt (exponential backoff)
                time.sleep(2 ** attempt)

        # Nem kellene ide eljutni, de biztonsági háló
        duration_ms = int((time.time() - start_time) * 1000)
        return {
            "success": False,
            "response": "",
            "error": "Max retries exceeded",
            "metadata": {
                "duration_ms": duration_ms,
                "response_length": 0,
                "streaming": stream,
                "attempt": self.max_retries
            },
            "performance": {
                "total_duration_ms": duration_ms,
                "api_latency_ms": None,
                "ttft_ms": None,
                "response_tokens": 0,
                "tokens_per_second": 0,
                "retry_count": self.max_retries
            }
        }

    def _send_streaming_request(self, payload: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        """
        Streaming POST request küldése performance tracking-gel.

        Args:
            payload: Request payload

        Returns:
            Tuple of (response text, performance metrics dict)

        Raises:
            Exception: Ha a request sikertelen
        """
        request_start = time.time()

        response = requests.post(
            self.chat_endpoint,
            json=payload,
            timeout=self.timeout,
            stream=True,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream"
            }
        )

        # Státusz ellenőrzés
        if response.status_code != 200:
            error_text = response.text
            try:
                error_json = response.json()
                error_msg = error_json.get("message", error_text)
            except:
                error_msg = error_text
            raise Exception(f"HTTP {response.status_code}: {error_msg}")

        # Streaming response feldolgozása performance tracking-gel
        full_response = ""
        first_token_time = None
        token_count = 0

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')

                # Vercel AI SDK stream formátum: 0:"text"
                # Például: 0:"M", 0:"aug", 0:"li"
                if ':' in decoded_line:
                    try:
                        # Split első kettőspontnál
                        parts = decoded_line.split(':', 1)
                        if len(parts) == 2:
                            # Második rész JSON string: "M", "aug", stb.
                            content = json.loads(parts[1])
                            if isinstance(content, str):
                                # TTFT tracking: első token időpont
                                if first_token_time is None and content:
                                    first_token_time = time.time()

                                full_response += content
                                token_count += 1
                    except (json.JSONDecodeError, ValueError):
                        # Ha nem sikerül parse-olni, próbáljuk meg SSE formátumként
                        pass

                # Server-Sent Events formátum fallback: "data: {...}"
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]  # "data: " prefix levágása

                    # Skip [DONE] marker
                    if data_str.strip() == '[DONE]':
                        break

                    try:
                        data = json.loads(data_str)

                        # Vercel AI SDK formátum: {"type":"text","value":"..."}
                        if isinstance(data, dict):
                            if 'type' in data and data['type'] == 'text':
                                content = data.get('value', '')
                                if first_token_time is None and content:
                                    first_token_time = time.time()
                                full_response += content
                                token_count += 1
                            # OpenAI API formátum
                            elif 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if first_token_time is None and content:
                                    first_token_time = time.time()
                                full_response += content
                                token_count += 1
                            # Egyszerűsített formátum
                            elif 'content' in data:
                                if first_token_time is None and data['content']:
                                    first_token_time = time.time()
                                full_response += data['content']
                                token_count += 1
                    except json.JSONDecodeError:
                        continue

        # Performance metrics
        perf_metrics = {}
        if first_token_time:
            ttft_ms = int((first_token_time - request_start) * 1000)
            perf_metrics["ttft_ms"] = ttft_ms

        perf_metrics["stream_tokens"] = token_count

        return full_response.strip(), perf_metrics

    def _send_regular_request(self, payload: Dict[str, Any]) -> str:
        """
        Nem-streaming POST request küldése.

        Args:
            payload: Request payload

        Returns:
            Response text

        Raises:
            Exception: Ha a request sikertelen
        """
        response = requests.post(
            self.chat_endpoint,
            json=payload,
            timeout=self.timeout,
            headers={"Content-Type": "application/json"}
        )

        # Státusz ellenőrzés
        if response.status_code != 200:
            error_text = response.text
            try:
                error_json = response.json()
                error_msg = error_json.get("message", error_text)
            except:
                error_msg = error_text
            raise Exception(f"HTTP {response.status_code}: {error_msg}")

        # Response feldolgozása
        try:
            result = response.json()
            return result.get("response", "")
        except json.JSONDecodeError:
            return response.text

    def send_conversation(
        self,
        conversation: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Teljes beszélgetés elküldése (multi-turn support).

        Args:
            conversation: Beszélgetés lista
                [
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                    {"role": "user", "content": "..."}
                ]

        Returns:
            Dict az eredménnyel (ugyanaz, mint send_message)

        Példa:
            >>> client = AssistantClient()
            >>> conversation = [
            ...     {"role": "user", "content": "Ki az a Maugli?"},
            ...     {"role": "assistant", "content": "Maugli egy emberi gyerek..."},
            ...     {"role": "user", "content": "És ki nevelte fel?"}
            ... ]
            >>> result = client.send_conversation(conversation)
        """
        return self.send_message(conversation)

    def health_check(self) -> bool:
        """
        Backend health check.

        Returns:
            True ha a backend elérhető, False egyébként
        """
        try:
            response = requests.get(
                f"{self.base_url}/",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_message(role: str, content: str) -> Dict[str, str]:
    """
    Üzenet objektum létrehozása.

    Args:
        role: "user" vagy "assistant"
        content: Üzenet tartalma

    Returns:
        Message dict
    """
    if role not in ["user", "assistant"]:
        raise ValueError(f"Invalid role: {role}. Must be 'user' or 'assistant'")

    return {
        "role": role,
        "content": content
    }


def format_conversation_history(
    conversation: List[Dict[str, str]]
) -> str:
    """
    Beszélgetés formázása emberbarát formátumba.

    Args:
        conversation: Beszélgetés lista

    Returns:
        Formázott string
    """
    lines = []
    for msg in conversation:
        role_label = "👤 User:" if msg["role"] == "user" else "🤖 Assistant:"
        lines.append(f"{role_label} {msg['content']}")

    return "\n".join(lines)


# ============================================================================
# TESTING & EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=== RAG Assistant Client Test ===\n")

    # Kliens létrehozása
    client = AssistantClient()

    # Health check
    print("1. Health check...")
    if client.health_check():
        print("   ✓ Backend is running\n")
    else:
        print("   ✗ Backend is NOT running!")
        print("   Please start the backend: cd assistant && npm run dev")
        exit(1)

    # Single turn test
    print("2. Single turn test...")
    messages = [
        create_message("user", "Ki az a Maugli?")
    ]

    result = client.send_message(messages)

    if result["success"]:
        print(f"   ✓ Success! (Duration: {result['metadata']['duration_ms']}ms)")
        print(f"   Response: {result['response'][:100]}...")
    else:
        print(f"   ✗ Failed: {result['error']}")

    print()

    # Multi-turn test
    print("3. Multi-turn test...")
    conversation = [
        create_message("user", "Ki az a Balú?"),
        create_message("assistant", "Balú egy medve, aki Maugli egyik tanítója a dzsungelben."),
        create_message("user", "Mit tanít neki?")
    ]

    result = client.send_conversation(conversation)

    if result["success"]:
        print(f"   ✓ Success! (Duration: {result['metadata']['duration_ms']}ms)")
        print(f"   Response: {result['response'][:100]}...")
    else:
        print(f"   ✗ Failed: {result['error']}")

    print("\n=== Test Complete ===")
