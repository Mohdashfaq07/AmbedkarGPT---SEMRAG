"""
ollama_client.py
Simplified Ollama wrapper that returns plain text response
"""

import subprocess

class OllamaClient:
    def __init__(self, model="mistral"):
        self.model = model

    def generate(self, prompt: str) -> str:
        process = subprocess.Popen(
            ["ollama", "run", self.model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",   # enforce UTF-8 decoding
            errors="ignore"     # ignore invalid characters instead of crashing
        )

        stdout, stderr = process.communicate(prompt)

        if stderr:
            print("Ollama stderr:", stderr)

        return stdout.strip()
