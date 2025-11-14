
const express = require("express");
const app = express();

// Middleware to parse JSON bodies
app.use(express.json());

// ----------------------------
// Logging Middleware
// ----------------------------
app.use((req, res, next) => {
  req.startTime = Date.now();

  console.log(
    `[REQUEST] Method: ${req.method} | Path: ${req.path} | IP: ${req.ip} | Body: ${JSON.stringify(req.body)}`
  );

  // Capture send() to log response
  const originalSend = res.send;
  res.send = function (data) {
    const duration = (Date.now() - req.startTime) / 1000;
    console.log(
      `[RESPONSE] Status: ${res.statusCode} | Duration: ${duration}s | Path: ${req.path}`
    );
    return originalSend.apply(res, arguments);
  };

  next();
});

// ----------------------------
// ROUTES
// ----------------------------

// Home route
app.get("/", (req, res) => {
  res.json({ message: "Hello, World!" });
});

// Heavy compute route (same CPU load as Flask)
app.get("/compute", (req, res) => {
  let x = 0;
  for (let i = 0; i < 20000; i++) {
    x += i * i;
  }
  res.json({ result: x });
});

// ----------------------------
// START SERVER
// ----------------------------
const PORT = 5000;
app.listen(PORT, () => {
  console.log(`Node.js server running on http://localhost:${PORT}`);
});
