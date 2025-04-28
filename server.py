import subprocess
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import os


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
                query = json.loads(post_data.decode("utf-8"))["query"]
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid JSON data")
                return

            try:
                # Define paths
                venv_python = "python"
                cli_path = "./cli.py"

                # Verify paths exist
                if not os.path.exists(cli_path):
                    raise FileNotFoundError(f"cli.py not found at {cli_path}")

                # Run cli.py query command
                cmd = [venv_python, cli_path, "query", query, "--mode", "docs"]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                if result.returncode == 0:
                    # Filter output to include only lines after "Response:"
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
                    else:
                        response_text = result.stdout.strip()
                    response = {"result": response_text}
                else:
                    response = {"error": result.stderr}
            except subprocess.TimeoutExpired:
                response = {"error": "Query timed out after 60 seconds."}
            except FileNotFoundError as e:
                response = {"error": str(e)}
            except Exception as e:
                response = {"error": f"Unexpected error: {str(e)}"}

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
