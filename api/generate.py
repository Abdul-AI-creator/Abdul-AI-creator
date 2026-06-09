"""Vercel serverless API endpoint for LeadWin RAG."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler
from typing import Any

from leadwin_rag.service import generate_response_payload


MAX_BODY_BYTES = 4_500_000


class handler(BaseHTTPRequestHandler):
    """Handle browser requests from the Vercel static UI."""

    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("content-length", "0"))
            if length <= 0:
                self._send_json({"error": "Request body is required"}, status=400)
                return
            if length > MAX_BODY_BYTES:
                self._send_json({"error": "Request body is too large"}, status=413)
                return

            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            knowledge_documents = self._extract_knowledge(payload)
            leads_csv = str(payload.get("leads_csv", "")).strip()
            top_k = int(payload.get("top_k", 4))

            if not knowledge_documents:
                self._send_json({"error": "At least one knowledge document is required"}, status=400)
                return
            if not leads_csv:
                self._send_json({"error": "Leads CSV text is required"}, status=400)
                return

            result = generate_response_payload(
                knowledge_documents=knowledge_documents,
                leads_csv=leads_csv,
                top_k=max(1, min(8, top_k)),
            )
            self._send_json(result)
        except json.JSONDecodeError:
            self._send_json({"error": "Request body must be valid JSON"}, status=400)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # pragma: no cover - defensive boundary for serverless runtime
            self._send_json({"error": f"Unexpected server error: {exc}"}, status=500)

    def _extract_knowledge(self, payload: dict[str, Any]) -> list[tuple[str, str]]:
        documents = []
        for idx, item in enumerate(payload.get("knowledge", []), start=1):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or f"knowledge-{idx}.txt")
            content = str(item.get("content") or "").strip()
            if content:
                documents.append((name, content))
        return documents

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
