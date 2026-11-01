
const mongoose = require("mongoose");

const scanSchema = new mongoose.Schema(
  {
    imagePath: {
      type: String,
      required: true,
    },
    status: {
      type: String,
      enum: ["received", "processing", "completed", "failed"],
      default: "received",
    },
    analysisType: {
      type: String,
      default: "camera-research",
    },
    analysisResult: {
  type: mongoose.Schema.Types.Mixed,
  default: null
},
  },
  { timestamps: true }
);

module.exports =
  mongoose.models.Scan || mongoose.model("Scan", scanSchema);
  
