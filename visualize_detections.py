import sys
import requests
import json
from PIL import Image, ImageDraw, ImageFont

BASE_URL = "http://127.0.0.1:8001"

def visualize(image_path, output_path):
    print(f"Sending {image_path} to API...")
    with open(image_path, 'rb') as f:
        r = requests.post(
            f"{BASE_URL}/api/v1/detect",
            params={"confidence_threshold": 0.25},
            files={"file": (image_path, f, "image/jpeg")}
        )
    
    if r.status_code != 200:
        print(f"API Error: {r.status_code} - {r.text}")
        return
        
    result = r.json()
    detections = result.get("detections", [])
    print(f"Got {len(detections)} detections.")
    
    # Draw on image
    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        # Colors for different classes
        colors = {
            "Person": "blue",
            "Hardhat": "green",
            "Safety Vest": "orange",
            "NO-Hardhat": "red",
            "NO-Safety Vest": "red"
        }
        
        for d in detections:
            cls_name = d["class_name"]
            conf = d["confidence"]
            bbox = d["bbox"]
            
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            color = colors.get(cls_name, "magenta")
            
            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            
            # Draw label
            label = f"{cls_name} {conf:.2f}"
            draw.rectangle([x1, y1-20, x1 + len(label)*7, y1], fill=color)
            draw.text((x1+2, y1-18), label, fill="white")
            
        img.save(output_path)
        print(f"Saved visualization to {output_path}")
        
    except Exception as e:
        print(f"Failed to draw: {e}")

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_construction.jpg"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "output_viz.jpg"
    visualize(image_path, out_path)
