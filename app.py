import time
import logging
from flask import Flask, request, jsonify

# -------------------------
# Configure Console Logging
# -------------------------
logger = logging.getLogger("flask-logger")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
))
logger.addHandler(console_handler)

app = Flask(__name__)

# -------------------------
# Log every request BEFORE it executes
# -------------------------
@app.before_request
def start_timer():
    request.start_time = time.time()

@app.before_request
def log_request_info():
    logger.info(
        f"[REQUEST] Method: {request.method} | "
        f"Path: {request.path} | "
        f"Args: {dict(request.args)} | "
        f"IP: {request.remote_addr} | "
        f"Body: {request.get_data(as_text=True)}"
    )

# -------------------------
# Log every response AFTER execution
# -------------------------
@app.after_request
def log_response_info(response):
    duration = round(time.time() - request.start_time, 4)
    logger.info(
        f"[RESPONSE] Status: {response.status_code} | "
        f"Duration: {duration}s | "
        f"Path: {request.path}"
    )
    return response


# -------------------------
# Example Routes
# -------------------------
@app.route("/")
def home():
    return jsonify({"message": "Hello, World!"})

@app.route("/compute")
def compute():
    x = sum(i*i for i in range(20000))
    return jsonify({"result": x})


# -------------------------
# Run Flask Server
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
