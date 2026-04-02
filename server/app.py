"""
server.py — OpenEnv HTTP server entrypoint
Run with:  uv run server
"""
import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from environment import SnakeEnv

# Global env instance
env = SnakeEnv(grid_size=10, max_steps=500)

def numpy_to_python(obj):
    """Recursively convert numpy types for JSON serialisation."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_python(i) for i in obj]
    return obj


class EnvHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(numpy_to_python(data)).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}

    # ---------- Routes ----------

    def do_GET(self):
        if self.path == "/" or self.path == "":
            html = """<!DOCTYPE html>
<html>
<head>
    <title>🐍 SnakeEnv — OpenEnv Server</title>
    <style>
        body { font-family: Arial, sans-serif; background: #111; color: #eee; padding: 40px; }
        h1 { color: #00ff88; }
        .card { background: #1e1e1e; border-radius: 10px; padding: 20px; margin: 16px 0; }
        .badge { display: inline-block; background: #00ff88; color: #000; border-radius: 4px; padding: 2px 8px; font-size: 12px; margin-right: 8px; font-weight: bold; }
        .badge.post { background: #ff9900; }
        code { background: #333; padding: 2px 8px; border-radius: 4px; font-size: 14px; }
        button { background: #00ff88; color: #000; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; font-weight: bold; margin: 8px 4px; }
        button:hover { background: #00cc66; }
        #output { background: #000; color: #00ff88; padding: 16px; border-radius: 8px; font-family: monospace; white-space: pre; min-height: 80px; margin-top: 16px; }
        .status { color: #00ff88; font-weight: bold; }
    </style>
</head>
<body>
    <h1>🐍 SnakeEnv is Running!</h1>
    <p class="status">✅ Server is live at http://localhost:7860</p>

    <div class="card">
        <h3>Available Endpoints</h3>
        <p><span class="badge">GET</span> <code>/health</code> — Check server status</p>
        <p><span class="badge">GET</span> <code>/info</code> — Environment metadata</p>
        <p><span class="badge post">POST</span> <code>/reset</code> — Reset environment</p>
        <p><span class="badge post">POST</span> <code>/step</code> — Take action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)</p>
        <p><span class="badge post">POST</span> <code>/render</code> — Render current state</p>
    </div>

    <div class="card">
        <h3>Try it out</h3>
        <button onclick="callReset()">🔄 Reset Game</button>
        <button onclick="callStep(0)">⬆️ UP</button>
        <button onclick="callStep(1)">⬇️ DOWN</button>
        <button onclick="callStep(2)">⬅️ LEFT</button>
        <button onclick="callStep(3)">➡️ RIGHT</button>
        <button onclick="callRender()">🎨 Render</button>
        <div id="output">Click a button to interact with the Snake environment...</div>
    </div>

    <script>
        async function callReset() {
            const res = await fetch('/reset', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({seed: 42})});
            const data = await res.json();
            document.getElementById('output').textContent = "RESET\\nScore: " + data.info.score + "  Length: " + data.info.snake_length + "  Steps: " + data.info.steps + "\\nHead: " + JSON.stringify(data.info.head_pos) + "  Food: " + JSON.stringify(data.info.food_pos);
        }
        async function callStep(action) {
            const labels = ['UP','DOWN','LEFT','RIGHT'];
            const res = await fetch('/step', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({action})});
            const data = await res.json();
            document.getElementById('output').textContent = "ACTION: " + labels[action] + "\\nReward: " + data.reward.toFixed(2) + "  Terminated: " + data.terminated + "  Truncated: " + data.truncated + "\\nScore: " + data.info.score + "  Length: " + data.info.snake_length + "  Steps: " + data.info.steps;
        }
        async function callRender() {
            const res = await fetch('/render', {method:'POST', headers:{'Content-Type':'application/json'}, body: '{}'});
            const data = await res.json();
            document.getElementById('output').textContent = data.render;
        }
    </script>
</body>
</html>"""
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        elif self.path == "/health":
            self._send_json({"status": "ok", "env": "SnakeEnv"})

        elif self.path == "/info":
            self._send_json({
                "name":        "SnakeEnv",
                "description": "Classic Snake Game RL Environment",
                "version":     "1.0.0",
                "observation_space": {
                    "type":  "Box",
                    "shape": list(env.observation_space.shape),
                    "low":   int(env.observation_space.low.min()),
                    "high":  int(env.observation_space.high.max()),
                    "dtype": str(env.observation_space.dtype),
                },
                "action_space": {
                    "type": "Discrete",
                    "n":    int(env.action_space.n),
                    "actions": {"0": "UP", "1": "DOWN", "2": "LEFT", "3": "RIGHT"},
                },
                "reward_range": [-10, 10],
                "max_steps":    env.max_steps,
            })
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        data = self._read_json()

        if self.path == "/reset":
            seed = data.get("seed", None)
            obs, info = env.reset(seed=seed)
            self._send_json({"observation": obs, "info": info})

        elif self.path == "/step":
            action = data.get("action")
            if action is None:
                self._send_json({"error": "action required"}, 400)
                return
            obs, reward, terminated, truncated, info = env.step(int(action))
            self._send_json({
                "observation": obs,
                "reward":      reward,
                "terminated":  terminated,
                "truncated":   truncated,
                "info":        info,
            })

        elif self.path == "/render":
            env_copy = SnakeEnv(grid_size=env.grid_size, render_mode="ansi")
            env_copy.snake     = list(env.snake)
            env_copy.food      = env.food
            env_copy.direction = env.direction
            env_copy.steps     = env.steps
            env_copy.score     = env.score
            env_copy._get_obs()   # populate grid
            rendered = env_copy._render_ansi()
            self._send_json({"render": rendered})

        else:
            self._send_json({"error": "Not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


if __name__ == "__main__":
    port = 7860
    print(f"🐍 SnakeEnv server running at http://localhost:{port}")
    print(f"   GET  /health  — health check")
    print(f"   GET  /info    — env metadata")
    print(f"   POST /reset   — reset environment")
    print(f"   POST /step    — take an action {{action: 0-3}}")
    print(f"   POST /render  — render current state")
    HTTPServer(("0.0.0.0", port), EnvHandler).serve_forever()
