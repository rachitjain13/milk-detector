
const express = require("express");
const cors = require("cors");
const dotenv = require("dotenv");
const mongoose = require("mongoose");
const multer = require("multer");
const path = require("path");
const fs = require("fs");
const crypto = require("crypto");
const jwt = require("jsonwebtoken");
const bcrypt = require("bcryptjs");
const { execFile } = require("child_process");
const util = require("util");
const requireAdmin = require("./middleware/requireAdmin");

const execFileAsync = util.promisify(execFile);
function getPythonPath() {
  if (process.env.PYTHON_BIN) {
    return process.env.PYTHON_BIN;
  }

  if (process.platform === "win32") {
    return path.join(
      __dirname,
      "..",
      "ai-engine",
      ".venv-deploy",
      "Scripts",
      "python.exe"
    );
  }

  return "python3";
}
dotenv.config();

const app = express();
const PORT = process.env.PORT || 5000;
const uploadDir = path.join(__dirname, "uploads");

fs.mkdirSync(uploadDir, { recursive: true });

app.use(cors({ origin: "http://localhost:5173" }));
app.use(express.json());

// ---------- Python image analyzer ----------


// ---------- Python image analyzer ----------

async function analyzeMilkImage(imagePath) {
  const projectRoot = path.join(__dirname, "..");
const pythonPath = getPythonPath();

  const predictorPath = path.join(
    projectRoot,
    "ai-engine",
    "milkoptic_predict.py"
  );

  if (!fs.existsSync(pythonPath)) {
    throw new Error(`Python not found: ${pythonPath}`);
  }

  if (!fs.existsSync(predictorPath)) {
    throw new Error(`Prediction script not found: ${predictorPath}`);
  }

  const { stdout } = await execFileAsync(
    pythonPath,
    [predictorPath, imagePath],
    {
      timeout: 60000,
      maxBuffer: 1024 * 1024,
      windowsHide: true
    }
  );

  const jsonStart = stdout.indexOf("{");

  if (jsonStart === -1) {
    throw new Error("Python prediction script returned no JSON.");
  }

  return JSON.parse(stdout.slice(jsonStart));
}


  // The Python script may print diagnostic lines before its JSON.

// ---------- Image upload configuration ----------

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, uploadDir);
  },

  filename: (_req, file, cb) => {
    const extensions = {
      "image/jpeg": ".jpg",
      "image/png": ".png",
      "image/webp": ".webp"
    };

    const extension = extensions[file.mimetype];

    if (!extension) {
      return cb(new Error("Unsupported image type."));
    }

    cb(null, `${crypto.randomUUID()}${extension}`);
  }
});

const upload = multer({
  storage,
  limits: { fileSize: 8 * 1024 * 1024 },

  fileFilter: (_req, file, cb) => {
    const allowedTypes = [
      "image/jpeg",
      "image/png",
      "image/webp"
    ];

    if (allowedTypes.includes(file.mimetype)) {
      return cb(null, true);
    }

    cb(new Error("Only JPEG, PNG and WebP images are allowed."));
  }
});

const validationUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 8 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const allowed = [
      "image/jpeg",
      "image/png",
      "image/webp"
    ];

    if (allowed.includes(file.mimetype)) {
      return cb(null, true);
    }

    cb(new Error("Only JPEG, PNG and WebP images are allowed."));
  }
});

app.post(
  "/api/scans/validate",
  validationUpload.single("image"),
  async (req, res) => {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        valid: false,
        message: "Please provide a photo."
      });
    }

    const tempPath = path.join(
      uploadDir,
      `${crypto.randomUUID()}-validation.jpg`
    );

    try {
      if (!req.file.buffer || req.file.buffer.length === 0) {
        return res.status(400).json({
          success: false,
          valid: false,
          message: "The uploaded image is empty."
        });
      }

      fs.writeFileSync(tempPath, req.file.buffer);

      const projectRoot = path.join(__dirname, "..");
const pythonPath = getPythonPath();
      const validatorPath = path.join(
        projectRoot, "ai-engine", "validate_capture.py"
      );

      const { stdout } = await execFileAsync(
        pythonPath,
        [validatorPath, tempPath],
        { timeout: 30000, windowsHide: true, maxBuffer: 1024 * 1024 }
      );

      const result = JSON.parse(stdout);

      return res.status(result.valid ? 200 : 422).json({
        success: result.valid,
        ...result
      });
    } catch (error) {
      console.error("Capture validation error:", error.message);
      return res.status(500).json({
        success: false,
        valid: false,
        message: "Could not validate this photo. Please try again."
      });
    } finally {
      if (fs.existsSync(tempPath)) {
        fs.unlinkSync(tempPath);
      }
    }
  }
);

// ---------- Health check ----------

app.get("/api/health", (_req, res) => {
  res.json({
    success: true,
    message: "MilkOptic AI Backend Running!"
  });
});

// ---------- Upload and analyze milk image ----------

app.post(
  "/api/scans",
  upload.single("image"),
  async (req, res) => {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        message: "Please upload a photo."
      });
    }

    if (mongoose.connection.readyState !== 1) {
      fs.unlink(req.file.path, () => { });

      return res.status(503).json({
        success: false,
        message: "Database is not connected."
      });
    }

    const Scan = require("./models/scan");
    let scan;

    try {
      scan = await Scan.create({
        imagePath: req.file.filename,
        status: "processing",
        analysisType: "camera-research"
      });
    } catch (error) {
      fs.unlink(req.file.path, () => { });

      console.error("Scan database error:", error.message);

      return res.status(500).json({
        success: false,
        message: "Could not save the scan."
      });
    }

    try {
      const analysisResult = await analyzeMilkImage(req.file.path);

      scan.analysisResult = analysisResult;
      scan.status = "completed";

      await scan.save();

      return res.status(201).json({
        success: true,
        message: "Image analysis completed.",
        scanId: scan._id,
        status: scan.status,
        analysisResult
      });
    } catch (error) {
      console.error("Milk image analysis error:", error.message);

      try {
        scan.status = "failed";
        scan.analysisResult = {
          analysis_status: "analysis_failed",
          message: "The image could not be analyzed."
        };

        await scan.save();
      } catch (saveError) {
        console.error(
          "Could not update failed scan:",
          saveError.message
        );
      }

      // Keep the uploaded image so the failed scan can be investigated.
      return res.status(202).json({
        success: false,
        message: "Photo received, but image analysis failed.",
        scanId: scan._id,
        status: "failed"
      });
    }
  }
);

// ---------- Admin login ----------

app.post("/api/admin/login", async (req, res) => {
  const { username, password } = req.body || {};

  if (
    !process.env.ADMIN_USERNAME ||
    !process.env.ADMIN_PASSWORD_HASH ||
    !process.env.JWT_SECRET
  ) {
    return res.status(503).json({
      success: false,
      message: "Admin credentials are not configured."
    });
  }

  const validUsername = username === process.env.ADMIN_USERNAME;

  const validPassword =
    typeof password === "string" &&
    await bcrypt.compare(
      password,
      process.env.ADMIN_PASSWORD_HASH
    );

  if (!validUsername || !validPassword) {
    return res.status(401).json({
      success: false,
      message: "Invalid username or password."
    });
  }

  const token = jwt.sign(
    { role: "admin" },
    process.env.JWT_SECRET,
    { expiresIn: "1h" }
  );

  res.json({
    success: true,
    token,
    expiresIn: 3600
  });
});

// ---------- Admin scan list ----------

app.get("/api/admin/scans", requireAdmin, async (_req, res) => {
  try {
    const Scan = require("./models/scan");

    const scans = await Scan.find()
      .select("_id status analysisType createdAt")
      .sort({ createdAt: -1 })
      .limit(100)
      .lean();

    const total = await Scan.countDocuments();

    res.json({
      success: true,
      total,
      scans
    });
  } catch (error) {
    console.error("Admin scan list error:", error.message);

    res.status(500).json({
      success: false,
      message: "Could not load scans."
    });
  }
});

// ---------- Error handler ----------

app.use((err, _req, res, _next) => {
  console.error("Request error:", err.message);

  if (err instanceof multer.MulterError) {
    return res.status(400).json({
      success: false,
      message:
        err.code === "LIMIT_FILE_SIZE"
          ? "Image must be 8 MB or smaller."
          : err.message
    });
  }

  res.status(400).json({
    success: false,
    message: err.message || "Upload failed."
  });
});

// ---------- Start server ----------

async function startServer() {
  try {
    if (!process.env.MONGODB_URI) {
      throw new Error("MONGODB_URI is missing from .env");
    }

    await mongoose.connect(process.env.MONGODB_URI);

    console.log("MongoDB connected");

    app.listen(PORT, () => {
      console.log(`Server running at http://localhost:${PORT}`);
    });

  } catch (error) {
    console.error("Capture validation error:", error.message);
    if (error.stderr) {
      console.error("Python stderr:", error.stderr);
    }
    if (error.stdout) {
      console.error("Python stdout:", error.stdout);
    }

    return res.status(500).json({
      success: false,
      valid: false,
      message: "Could not validate this photo. Check the backend terminal for details."
    });
  }
}

startServer();
