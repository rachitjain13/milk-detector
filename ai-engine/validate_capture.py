
import sys
import json
import traceback

from analyze_image import validate_capture


if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            raise ValueError("Image path is missing.")

        result = validate_capture(sys.argv[1])
        print(json.dumps(result))

    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({
            "valid": False,
            "message": f"Validation failed: {exc}"
        }))
