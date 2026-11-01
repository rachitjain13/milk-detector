import API_BASE_URL from "../services/apiConfig";
import { useEffect, useRef, useState } from "react";
import {
  FaceDetector,
  FilesetResolver,
} from "@mediapipe/tasks-vision";

type AnalysisResult = {
  image_width?: number;
  image_height?: number;
  brightness_mean?: number;
  sharpness_estimate?: number;
  quality_status?: string;
  quality_warnings?: string[];
  analysis_status?: string;
  adulterant_identified?: boolean;
  milk_region_detected?: boolean;
  region_review_required?: boolean;
  candidate_region?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

type ApiResponse = {
  success: boolean;
  valid?: boolean;
  message: string;
  scanId?: string;
  status?: string;
  analysisResult?: AnalysisResult;
};

type FaceStatus = "idle" | "loading" | "clear" | "detected" | "error";

function MilkCamera() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const faceDetectorRef = useRef<FaceDetector | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const faceStatusRef = useRef<FaceStatus>("idle");

  const [cameraOn, setCameraOn] = useState(false);
  const [photo, setPhoto] = useState<string | null>(null);
  const [confirmedMilk, setConfirmedMilk] = useState(false);
  const [faceStatus, setFaceStatus] = useState<FaceStatus>("idle");

  const [validating, setValidating] = useState(false);
  const [validated, setValidated] = useState(false);
  const [uploading, setUploading] = useState(false);

  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [scanId, setScanId] = useState("");
  const [scanStatus, setScanStatus] = useState("");
  const [analysis, setAnalysis] =
    useState<AnalysisResult | null>(null);

  const busy = validating || uploading;

  function updateFaceStatus(status: FaceStatus) {
    faceStatusRef.current = status;
    setFaceStatus(status);
  }

  async function startCamera() {
    setError("");
    setMessage("");
    setFaceStatus("loading");
    faceStatusRef.current = "loading";

    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Camera unavailable. Use HTTPS or localhost.");
      updateFaceStatus("error");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraOn(true);
    } catch (err) {
      console.error("Camera startup failed:", err);
      setError("Allow camera permission and try again.");
      updateFaceStatus("error");
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;

    setCameraOn(false);
    updateFaceStatus("idle");
  }

  // Initialize MediaPipe and continuously check the camera for faces.
  useEffect(() => {
    if (!cameraOn) return;

    let cancelled = false;
    let detector: FaceDetector | null = null;
    let lastVideoTime = -1;

    async function initializeFaceDetector() {
      try {
        updateFaceStatus("loading");

        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
        );

        if (cancelled) return;

        detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath:
              "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            delegate: "CPU",
          },
          runningMode: "VIDEO",
          minDetectionConfidence: 0.5,
        });

        if (cancelled) {
          detector.close();
          detector = null;
          return;
        }

        faceDetectorRef.current = detector;

        detectFaces();
      } catch (err) {
        console.error("MediaPipe initialization failed:", err);

        if (!cancelled) {
          updateFaceStatus("error");
          setError(
            "Face detection could not start. Check your internet connection and restart the camera."
          );
        }
      }
    }

    function detectFaces() {
      if (cancelled) return;

      const video = videoRef.current;
      const faceDetector = faceDetectorRef.current;

      if (
        video &&
        faceDetector &&
        video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA &&
        video.currentTime !== lastVideoTime
      ) {
        try {
          lastVideoTime = video.currentTime;

          const result = faceDetector.detectForVideo(
            video,
            performance.now()
          );

          const faceFound = result.detections.length > 0;

          const nextStatus: FaceStatus = faceFound
            ? "detected"
            : "clear";

          if (faceStatusRef.current !== nextStatus) {
            updateFaceStatus(nextStatus);
          }
        } catch (err) {
          console.error("Live face detection failed:", err);

          if (!cancelled) {
            updateFaceStatus("error");
            setError(
              "Live face detection failed. Please restart the camera."
            );
          }

          return;
        }
      }

      if (!cancelled) {
        animationFrameRef.current =
          requestAnimationFrame(detectFaces);
      }
    }

    void initializeFaceDetector();

    return () => {
      cancelled = true;

      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }

      faceDetectorRef.current = null;

      if (detector) {
        detector.close();
        detector = null;
      }
    };
  }, [cameraOn]);

  function capturePhoto() {
    setError("");
    setMessage("");

    if (faceStatusRef.current !== "clear") {
      setError(
        faceStatusRef.current === "detected"
          ? "A face was detected. Keep only the milk glass in the frame."
          : "Face detection is not ready. Please wait and try again."
      );
      return;
    }

    if (!confirmedMilk) {
      setError(
        "Please confirm that a glass containing milk is in the frame."
      );
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas || !video.videoWidth) {
      setError("Camera is not ready. Please try again.");
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    if (!context) {
      setError("Could not capture the image.");
      return;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    setPhoto(canvas.toDataURL("image/jpeg", 0.9));
    setValidated(false);
    setAnalysis(null);
    setScanId("");
    setScanStatus("");

    stopCamera();
  }

  function retakePhoto() {
    setPhoto(null);
    setConfirmedMilk(false);
    setValidated(false);
    setAnalysis(null);
    setScanId("");
    setScanStatus("");
    setError("");
    setMessage("");

    void startCamera();
  }

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    };
  }, []);

  async function validatePhoto() {
    if (!photo || busy) return;

    setValidating(true);
    setError("");
    setMessage("");
    setValidated(false);
    setAnalysis(null);
    setScanId("");
    setScanStatus("");

    try {
      const blob = await fetch(photo).then((response) =>
        response.blob()
      );

      const formData = new FormData();
      formData.append("image", blob, "milk-sample.jpg");

      const response = await fetch(
        `${API_BASE_URL}/api/scans/validate`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data: ApiResponse = await response.json();

      if (!response.ok || data.valid !== true) {
        throw new Error(data.message || "Photo validation failed.");
      }

      setValidated(true);
      setMessage(
        "Basic image checks passed. Glass and milk presence are not verified."
      );
    } catch (err) {
      setValidated(false);

      setError(
        err instanceof Error
          ? err.message
          : "Could not validate the photo."
      );
    } finally {
      setValidating(false);
    }
  }

  async function uploadPhoto() {
    if (!photo || !validated || busy) return;

    setUploading(true);
    setError("");
    setMessage("");
    setAnalysis(null);
    setScanId("");
    setScanStatus("processing");

    try {
      const blob = await fetch(photo).then((response) =>
        response.blob()
      );

      const formData = new FormData();
      formData.append("image", blob, "milk-sample.jpg");

      const response = await fetch(
        `${API_BASE_URL}/api/scans`,
        {
          method: "POST",
          body: formData,
        }
      );

      const data: ApiResponse = await response.json();

      if (data.scanId) setScanId(data.scanId);
      if (data.status) setScanStatus(data.status);
      if (data.analysisResult) setAnalysis(data.analysisResult);

      if (!response.ok && response.status !== 202) {
        throw new Error(data.message || "Upload failed.");
      }

      if (!data.success || data.status === "failed") {
        setError(data.message || "Image analysis failed.");
      } else {
        setMessage(data.message || "Image analysis completed.");
      }
    } catch (err) {
      setScanStatus("failed");

      setError(
        err instanceof Error
          ? err.message
          : "Upload failed. Try again."
      );
    } finally {
      setUploading(false);
    }
  }

  return (
    <section className="camera-card" id="milk-scanner">
      <h2>Scan Your Milk</h2>

      <p>
        Capture a clear photo of a transparent glass containing milk.
      </p>

      {error && (
        <p className="camera-error" role="alert">
          {error}
        </p>
      )}

      {message && (
        <p className="camera-success" role="status">
          {message}
        </p>
      )}

      {!photo && (
        <div className="capture-guidance">
          <strong>Before capturing</strong>

          <ul>
            <li>Place the milk in a transparent glass.</li>
            <li>Keep the complete glass inside the frame.</li>
            <li>Use good lighting and avoid reflections.</li>
            <li>Do not capture a selfie or an empty glass.</li>
          </ul>
        </div>
      )}

      {!photo && (
        <video
          ref={videoRef}
          className="camera-preview"
          autoPlay
          playsInline
          muted
        />
      )}

      {photo && (
        <img
          src={photo}
          className="camera-preview"
          alt="Captured sample for validation"
        />
      )}

      <canvas ref={canvasRef} hidden />

      {!cameraOn && !photo && (
        <button className="scan-button" onClick={startCamera}>
          Open Camera
        </button>
      )}

      {cameraOn && (
        <>
          <div className="face-detection-status" role="status">
            {faceStatus === "loading" && (
              <p>⏳ Initializing face detection...</p>
            )}

            {faceStatus === "clear" && (
              <p>✅ No face detected. Face check passed.</p>
            )}

            {faceStatus === "detected" && (
              <p>
                ⚠️ Face detected! Keep only the milk glass in the frame.
              </p>
            )}

            {faceStatus === "error" && (
              <p>
                ❌ Face detection unavailable. Capture is blocked.
              </p>
            )}

            {faceStatus === "idle" && (
              <p>Waiting for camera...</p>
            )}
          </div>

          <label className="milk-confirmation">
            <input
              type="checkbox"
              checked={confirmedMilk}
              onChange={(event) =>
                setConfirmedMilk(event.target.checked)
              }
            />

            <span>
              I have placed a glass containing milk in the frame.
            </span>
          </label>

          <button
            className="scan-button"
            onClick={capturePhoto}
            disabled={
              !confirmedMilk ||
              faceStatus !== "clear" ||
              busy
            }
          >
            Capture Photo
          </button>
        </>
      )}

      {photo && (
        <div className="camera-actions">
          <p>Photo captured. Validate it before uploading.</p>

          <button
            className="scan-button"
            onClick={retakePhoto}
            disabled={busy}
          >
            Retake Photo
          </button>

          <button
            className="scan-button"
            onClick={validatePhoto}
            disabled={busy || validated}
          >
            {validating
              ? "Validating..."
              : validated
                ? "Validation Passed"
                : "Validate Photo"}
          </button>

          <button
            className="scan-button"
            onClick={uploadPhoto}
            disabled={busy || !validated}
          >
            {uploading ? "Analyzing..." : "Upload for Analysis"}
          </button>

          {scanId && (
            <p>
              <strong>Scan ID:</strong> {scanId}
            </p>
          )}

          {scanStatus && (
            <p>
              <strong>Status:</strong> {scanStatus}
            </p>
          )}

          {analysis && (
            <div className="analysis-results">
              <h3>Image Analysis Results</h3>

              <p>
                <strong>Analysis:</strong>{" "}
                {analysis.analysis_status ?? "Not available"}
              </p>

              <p>
                <strong>Dimensions:</strong>{" "}
                {analysis.image_width ?? "—"} ×{" "}
                {analysis.image_height ?? "—"}
              </p>

              <p>
                <strong>Quality:</strong>{" "}
                {analysis.quality_status ?? "Not available"}
              </p>

              <p>
                <strong>Brightness:</strong>{" "}
                {analysis.brightness_mean ?? "—"}
              </p>

              <p>
                <strong>Sharpness:</strong>{" "}
                {analysis.sharpness_estimate ?? "—"}
              </p>

              <p>
                <strong>Milk verified:</strong>{" "}
                {analysis.milk_region_detected ? "Yes" : "No"}
              </p>

              {analysis.quality_warnings?.length ? (
                <>
                  <strong>Warnings</strong>

                  <ul>
                    {analysis.quality_warnings.map(
                      (warning, index) => (
                        <li key={index}>{warning}</li>
                      )
                    )}
                  </ul>
                </>
              ) : null}

              <p className="note">
                Experimental image analysis only. This does not confirm
                milk purity, detect adulterants, or estimate adulteration
                percentages.
              </p>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

export default MilkCamera;
