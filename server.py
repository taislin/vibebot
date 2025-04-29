import subprocess
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
from querying import memory  # Import memory from querying.py


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            try:
                with open("index.html", "rb") as f:
                    self.wfile.write(f.read())
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"index.html not found")
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/query":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode("utf-8"))
                query = data.get("query")
                mode = data.get("mode", "general")
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid JSON data")
                return

            try:
                # Run cli.py query command
                venv_python = (
                    os.path.join("venv", "Scripts", "python.exe")
                    if os.name == "nt"
                    else "/home/civ13/LLM-RAG-Bot/venv/bin/python"
                )
                cli_path = "cli.py"
                if not os.path.exists(venv_python):
                    raise FileNotFoundError(
                        f"Python executable not found at {venv_python}"
                    )
                if not os.path.exists(cli_path):
                    raise FileNotFoundError(f"cli.py not found at {cli_path}")

                cmd = [venv_python, cli_path, "query", query, "--mode", mode]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                if result.returncode == 0:
                    # Parse output
                    output_lines = result.stdout.splitlines()
                    response_start = next(
                        (
                            i
                            for i, line in enumerate(output_lines)
                            if line.startswith("Response:")
                        ),
                        -1,
                    )
                    if response_start != -1:
                        response_text = "\n".join(
                            output_lines[response_start + 1 :]
                        ).strip()
                        # Extract summary from response_text
                        summary_start = (
                            response_text.find("Summary:") + 8
                            if "Summary:" in response_text
                            else 0
                        )
                        response_text = (
                            response_text[summary_start:].strip()
                            if summary_start
                            else response_text
                        )
                    else:
                        response_text = result.stdout.strip()

                    # Load conversation history from memory
                    history = memory.load_memory_variables({})["history"]
                    history_formatted = [
                        {
                            "role": "user" if i % 2 == 0 else "assistant",
                            "content": msg.content,
                        }
                        for i, msg in enumerate(history)
                    ]

                    response = {"result": response_text, "history": history_formatted}
                else:
                    response = {"error": result.stderr, "history": []}
            except subprocess.TimeoutExpired:
                response = {
                    "error": "Query timed out after 60 seconds.",
                    "history": [],
                }
            except FileNotFoundError as e:
                response = {"error": str(e), "history": []}
            except Exception as e:
                response = {"error": f"Unexpected error: {str(e)}", "history": []}

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

        elif self.path == "/clear_memory":
            try:
                memory.clear()
                response = {"result": "Conversation memory cleared."}
            except Exception as e:
                response = {"error": f"Error clearing memory: {str(e)}"}

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode("utf-8"))

        else:
            self.send_response(404)
            self.end_headers()


def run(server_class=HTTPServer, handler_class=RequestHandler, port=8000):
    server_address = ("", port)
    httpd = server_class(server_address, handler_class)
    print(f"Starting server on http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    run()
