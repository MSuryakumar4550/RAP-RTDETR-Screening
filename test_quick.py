"""Quick test script for the running API server."""
import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8001"

def test_health():
    print("=" * 60)
    print("TEST 1: Health Check")
    print("=" * 60)
    r = requests.get(f"{BASE_URL}/health")
    print(json.dumps(r.json(), indent=2))
    print()

def test_detect(image_path, conf=0.25):
    print("=" * 60)
    print(f"TEST 2: Detection on '{image_path}' (conf={conf})")
    print("=" * 60)
    with open(image_path, 'rb') as f:
        r = requests.post(
            f"{BASE_URL}/api/v1/detect",
            params={"confidence_threshold": conf},
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    
    result = r.json()
    print(f"Status: {result['status']}")
    print(f"Inference Time: {result['inference_time_ms']} ms")
    print(f"Image Size: {result['image_width']} x {result['image_height']}")
    print(f"Total Detections: {result['total_detections']}")
    print()
    for d in result["detections"]:
        bbox = d["bbox"]
        print(f"  {d['class_name']:20s}  Conf: {d['confidence']:.3f}  "
              f"Box: [{bbox['x1']:.0f}, {bbox['y1']:.0f}, {bbox['x2']:.0f}, {bbox['y2']:.0f}]")
    print()
    return result

def test_reason(image_path, question):
    print("=" * 60)
    print(f"TEST 3: Reasoning - '{question}'")
    print("=" * 60)
    with open(image_path, 'rb') as f:
        r = requests.post(
            f"{BASE_URL}/api/v1/reason",
            files={"file": ("test.jpg", f, "image/jpeg")},
            data={"question": question}
        )
    
    result = r.json()
    print(json.dumps(result, indent=2))
    print()
    return result

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_construction.jpg"
    
    test_health()
    test_detect(image_path)
    test_reason(image_path, "Is anyone not wearing a helmet?")
    test_reason(image_path, "How many workers are in this image?")
    test_reason(image_path, "What is the capital of France?")
    
    print("All tests complete!")
