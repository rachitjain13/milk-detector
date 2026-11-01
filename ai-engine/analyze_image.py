
import sys
import json
import os
import cv2
import numpy as np


def detect_candidate_region(image):
    """Find a broad candidate region using simple visual cues."""
    height, width = image.shape[:2]

    x1, x2 = int(width * 0.22), int(width * 0.78)
    y1, y2 = int(height * 0.15), int(height * 0.82)

    search = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)

    mask = (
        (hsv[:, :, 1] < 65) &
        (hsv[:, :, 2] > 100)
    ).astype(np.uint8) * 255

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if count <= 1:
        return None

    label = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    area = int(stats[label, cv2.CC_STAT_AREA])

    if area < 0.01 * search.shape[0] * search.shape[1]:
        return None

    rx = int(stats[label, cv2.CC_STAT_LEFT])
    ry = int(stats[label, cv2.CC_STAT_TOP])
    rw = int(stats[label, cv2.CC_STAT_WIDTH])
    rh = int(stats[label, cv2.CC_STAT_HEIGHT])

    return {
        "x": x1 + rx,
        "y": y1 + ry,
        "width": rw,
        "height": rh,
        "candidate_area_pixels": area
    }


def check_capture_image(image):
    """
    Flag images containing a detectable human face.
    Face detection is only a warning signal; it does not
    determine whether milk is present.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cascade_path = (
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    face_detector = cv2.CascadeClassifier(cascade_path)

    if face_detector.empty():
        return {
            "face_detection_available": False,
            "face_detected": None,
            "capture_validation_status": "manual_review_required"
        }

    faces = face_detector.detectMultiScale(
    gray,
    scaleFactor=1.05,
    minNeighbors=3,
    minSize=(30, 30)
)

    face_detected = len(faces) > 0

    return {
        "face_detection_available": True,
        "face_detected": face_detected,
        "face_count": int(len(faces)),
        "capture_validation_status": (
            "manual_review_required"
            if face_detected
            else "milk_presence_unverified"
        )
    }

def validate_capture(image_path):
    """Preflight checks before a scan is saved."""
    image = cv2.imread(image_path)

    if image is None:
        return {
            "valid": False,
            "message": "This file is not a readable image. Retake the photo."
        }

    height, width = image.shape[:2]

    if width < 300 or height < 300:
        return {
            "valid": False,
            "message": "Image resolution is too low. Capture a clearer photo."
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < 40:
        return {
            "valid": False,
            "message": "The photo is too dark. Improve the lighting and retry."
        }

    if brightness > 245:
        return {
            "valid": False,
            "message": "The photo is overexposed. Reduce glare and retry."
        }

    if sharpness < 20:
        return {
            "valid": False,
            "message": "The photo is blurry. Hold the camera steady and retry."
        }

    face_check = check_capture_image(image)

    return {
        "valid": True,
        "message": "Basic image checks passed.",
        "face_detected": False,
        "milk_presence_verified": False,
        "glass_presence_verified": False
    }

def analyze_image(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not open image: " + image_path)

    height, width = image.shape[:2]
    capture_check = check_capture_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bright_fraction = float(np.mean(gray >= 250))

    warnings = []

    if brightness < 40:
        warnings.append("Image may be too dark")
    if brightness > 245:
        warnings.append("Image may be overexposed")
    if sharpness < 20:
        warnings.append("Image may be blurry")
    if width < 300 or height < 300:
        warnings.append("Image resolution is low")
    if bright_fraction > 0.15:
        warnings.append("Possible strong reflections or glare")

    output_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "analysis_output"
    )
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(image_path))[0]

    result = {
        "image_width": width,
        "image_height": height,
        "brightness_mean": round(brightness, 2),
        "sharpness_estimate": round(sharpness, 2),
        "bright_pixel_fraction": round(bright_fraction, 4),
        "quality_status": (
            "review_required" if warnings
            else "initial_checks_passed"
        ),
        "quality_warnings": warnings,
        "adulterant_identified": False
    }

    # Preview 1: broad candidate region.
    candidate = detect_candidate_region(image)
    candidate_preview = image.copy()

    if candidate is not None:
        x = candidate["x"]
        y = candidate["y"]
        w = candidate["width"]
        h = candidate["height"]

        cv2.rectangle(
            candidate_preview, (x, y), (x + w, y + h),
            (0, 255, 0), 3
        )

        crop = image[y:y + h, x:x + w]

        if crop.size > 0:
            crop_path = os.path.join(
                output_dir, base_name + "_candidate.png"
            )
            if not cv2.imwrite(crop_path, crop):
                raise RuntimeError("Could not save candidate crop")
            result["candidate_crop_path"] = crop_path

        result["candidate_region"] = candidate
    else:
        result["candidate_region"] = None
        result["candidate_detection_status"] = "not_found"

    candidate_preview_path = os.path.join(
        output_dir, base_name + "_detection_preview.png"
    )
    if not cv2.imwrite(candidate_preview_path, candidate_preview):
        raise RuntimeError("Could not save candidate preview")

    result["detection_preview_path"] = candidate_preview_path
    result.update(capture_check)

    # Preview 2: estimated liquid-body area for a centered glass.
    # This is a fixed proportional ROI, not automatic milk segmentation.
    # It is tuned only as a starting point for similarly framed images.
    mx1 = int(width * 0.29)
    mx2 = int(width * 0.71)
    my1 = int(height * 0.29)
    my2 = int(height * 0.79)

    milk_body_crop = image[my1:my2, mx1:mx2]

    if milk_body_crop.size == 0:
        result["milk_body_status"] = "region_invalid"
        result["analysis_status"] = "image_quality_only"
        return result

    body_preview = image.copy()

    # Orange rectangle marks the estimated liquid-body area.
    cv2.rectangle(
        body_preview, (mx1, my1), (mx2, my2),
        (0, 165, 255), 3
    )
    cv2.putText(
        body_preview,
        "ESTIMATED ROI - REVIEW REQUIRED",
        (max(5, mx1 - 20), max(25, my1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 100, 255),
        2,
        cv2.LINE_AA
    )

    body_crop_path = os.path.join(
        output_dir, base_name + "_milk_body_candidate.png"
    )
    body_preview_path = os.path.join(
        output_dir, base_name + "_milk_body_preview.png"
    )

    if not cv2.imwrite(body_crop_path, milk_body_crop):
        raise RuntimeError("Could not save estimated body crop")

    if not cv2.imwrite(body_preview_path, body_preview):
        raise RuntimeError("Could not save body preview")

    # Descriptive features of this estimated ROI only.
    body_gray = cv2.cvtColor(
        milk_body_crop, cv2.COLOR_BGR2GRAY
    )
    body_hsv = cv2.cvtColor(
        milk_body_crop, cv2.COLOR_BGR2HSV
    )
    b, g, r = cv2.split(milk_body_crop)

    result.update({
        "milk_body_status": "estimated_roi_not_verified",
        "milk_region_detected": False,
        "region_review_required": True,
        "milk_body_crop_path": body_crop_path,
        "milk_body_preview_path": body_preview_path,
        "milk_body_roi": {
            "x": mx1,
            "y": my1,
            "width": mx2 - mx1,
            "height": my2 - my1
        },
        "body_brightness_mean": round(
            float(np.mean(body_gray)), 2
        ),
        "body_brightness_std": round(
            float(np.std(body_gray)), 2
        ),
        "body_red_mean": round(float(np.mean(r)), 2),
        "body_green_mean": round(float(np.mean(g)), 2),
        "body_blue_mean": round(float(np.mean(b)), 2),
        "body_saturation_mean": round(
            float(np.mean(body_hsv[:, :, 1])), 2
        ),
        "body_contrast_estimate": round(
            float(
                np.percentile(body_gray, 95)
                - np.percentile(body_gray, 5)
            ), 2
        ),
        "analysis_status": "candidate_roi_features_only"
    })

    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_image.py path/to/image.png")
        sys.exit(1)

    try:
        output = analyze_image(sys.argv[1])
        print(json.dumps(output, indent=2))
    except Exception as error:
        print(json.dumps({"error": str(error)}))
        sys.exit(1)
