"""
Structured Spatial Reasoning Engine (Pure Python)
--------------------------------------------------
Performs geometric bounding box containment and relational logic
to answer natural-language queries about PPE compliance and scene composition.

Hard Constraint #1 Compliant: No LangChain, CrewAI, or external LLM frameworks.
"""

from collections import Counter
from typing import List, Dict, Any
from app.schemas.models import DetectionItem, BoundingBox

def compute_box_intersection_ratio(box_a: BoundingBox, box_b_region: tuple) -> float:
    """
    Computes the fraction of box_a that falls within a given region (x1, y1, x2, y2).
    Java Equivalent: Standard 2D rectangle intersection geometry.
    """
    rx1, ry1, rx2, ry2 = box_b_region
    ix1 = max(box_a.x1, rx1)
    iy1 = max(box_a.y1, ry1)
    ix2 = min(box_a.x2, rx2)
    iy2 = min(box_a.y2, ry2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    box_a_area = box_a.width * box_a.height
    return intersection_area / box_a_area if box_a_area > 0 else 0.0


class SpatialReasoningEngine:
    """Solves structured spatial relationships between workers and safety gear."""

    @staticmethod
    def analyze_scene(detections: List[DetectionItem]) -> Dict[str, Any]:
        """
        Groups detections and performs spatial association between workers,
        helmets (head region), and safety vests (torso region).
        """
        people = [d for d in detections if d.class_name.lower() in ["person"]]
        helmets = [d for d in detections if d.class_name.lower() in ["hardhat", "hard-hat", "helmet"]]
        no_helmets = [d for d in detections if d.class_name.lower() in ["no-hardhat"]]
        vests = [d for d in detections if d.class_name.lower() in ["safety vest", "safety-vest"]]
        no_vests = [d for d in detections if d.class_name.lower() in ["no-safety vest"]]

        worker_profiles = []

        for idx, person in enumerate(people, 1):
            p_box = person.bbox
            p_h = p_box.height

            # Head Region: Top 25% vertical segment of the person's bounding box
            head_region = (
                p_box.x1 - 5,
                p_box.y1 - 10,
                p_box.x2 + 5,
                p_box.y1 + (0.28 * p_h)
            )

            # Torso Region: 20% to 65% vertical segment
            torso_region = (
                p_box.x1 - 5,
                p_box.y1 + (0.18 * p_h),
                p_box.x2 + 5,
                p_box.y1 + (0.68 * p_h)
            )

            # Check for NO-Hardhat detection overlapping this person (direct evidence of non-compliance)
            has_no_helmet_flag = False
            for nh in no_helmets:
                nh_center_x = (nh.bbox.x1 + nh.bbox.x2) / 2.0
                nh_center_y = (nh.bbox.y1 + nh.bbox.y2) / 2.0
                if (head_region[0] <= nh_center_x <= head_region[2] and 
                    head_region[1] <= nh_center_y <= head_region[3]):
                    has_no_helmet_flag = True
                    break
                elif compute_box_intersection_ratio(nh.bbox, head_region) > 0.25:
                    has_no_helmet_flag = True
                    break

            has_helmet = False
            if not has_no_helmet_flag:
                for h in helmets:
                    h_center_x = (h.bbox.x1 + h.bbox.x2) / 2.0
                    h_center_y = (h.bbox.y1 + h.bbox.y2) / 2.0
                    # Check center point containment or substantial overlap
                    if (head_region[0] <= h_center_x <= head_region[2] and 
                        head_region[1] <= h_center_y <= head_region[3]):
                        has_helmet = True
                        break
                    elif compute_box_intersection_ratio(h.bbox, head_region) > 0.35:
                        has_helmet = True
                        break

            # Check for NO-Safety Vest detection overlapping this person
            has_no_vest_flag = False
            for nv in no_vests:
                nv_center_x = (nv.bbox.x1 + nv.bbox.x2) / 2.0
                nv_center_y = (nv.bbox.y1 + nv.bbox.y2) / 2.0
                if (torso_region[0] <= nv_center_x <= torso_region[2] and 
                    torso_region[1] <= nv_center_y <= torso_region[3]):
                    has_no_vest_flag = True
                    break
                elif compute_box_intersection_ratio(nv.bbox, torso_region) > 0.25:
                    has_no_vest_flag = True
                    break

            has_vest = False
            if not has_no_vest_flag:
                for v in vests:
                    v_center_x = (v.bbox.x1 + v.bbox.x2) / 2.0
                    v_center_y = (v.bbox.y1 + v.bbox.y2) / 2.0
                    if (torso_region[0] <= v_center_x <= torso_region[2] and 
                        torso_region[1] <= v_center_y <= torso_region[3]):
                        has_vest = True
                        break
                    elif compute_box_intersection_ratio(v.bbox, torso_region) > 0.35:
                        has_vest = True
                        break

            worker_profiles.append({
                "worker_id": idx,
                "bbox": p_box,
                "has_helmet": has_helmet,
                "has_no_helmet_flag": has_no_helmet_flag,
                "has_vest": has_vest,
                "has_no_vest_flag": has_no_vest_flag,
                "compliant": has_helmet and has_vest and not has_no_helmet_flag and not has_no_vest_flag
            })

        class_counts = Counter(d.class_name for d in detections)

        return {
            "total_objects": len(detections),
            "class_counts": dict(class_counts),
            "total_people": len(people),
            "total_helmets": len(helmets),
            "total_vests": len(vests),
            "worker_profiles": worker_profiles
        }

    def answer_query(self, query: str, detections: List[DetectionItem]) -> str:
        """
        Formulates a clear, professional plain-English answer based on spatial analysis.
        """
        analysis = self.analyze_scene(detections)
        q = query.lower()

        # Query Type 1: Counting people / workers
        if any(w in q for w in ["how many people", "how many workers", "count people", "count workers", "number of people"]):
            count = analysis["total_people"]
            return (
                f"There {'is' if count == 1 else 'are'} {count} worker{'s' if count != 1 else ''} "
                f"detected in this image."
            )

        # Query Type 2: Helmet Compliance ("Is anyone not wearing a helmet?", "Are all wearing helmets?")
        if "helmet" in q or "hard-hat" in q or "hardhat" in q:
            people_count = analysis["total_people"]
            if people_count == 0:
                helmets = analysis["total_helmets"]
                return f"No workers were identified, but {helmets} standalone safety helmet(s) were detected in the scene."

            missing_helmet_ids = [w["worker_id"] for w in analysis["worker_profiles"] if not w["has_helmet"]]
            if not missing_helmet_ids:
                return f"Yes, all {people_count} worker(s) in this image are wearing safety helmets."
            else:
                return (
                    f"Non-compliance detected: {len(missing_helmet_ids)} out of {people_count} worker(s) "
                    f"(Worker ID: {', '.join(map(str, missing_helmet_ids))}) {'is' if len(missing_helmet_ids) == 1 else 'are'} "
                    "not wearing a safety helmet."
                )

        # Query Type 3: Safety Vest Compliance
        if "vest" in q or "jacket" in q:
            people_count = analysis["total_people"]
            missing_vest_ids = [w["worker_id"] for w in analysis["worker_profiles"] if not w["has_vest"]]
            if not missing_vest_ids:
                return f"Yes, all {people_count} worker(s) are wearing high-visibility safety vests."
            else:
                return (
                    f"Warning: {len(missing_vest_ids)} worker(s) (Worker ID: {', '.join(map(str, missing_vest_ids))}) "
                    "are observed without high-visibility safety vests."
                )

        # Query Type 4: Most common object
        if "most common" in q or "frequent" in q or "majority" in q:
            if not analysis["class_counts"]:
                return "No objects were detected."
            most_common_item, freq = Counter(analysis["class_counts"]).most_common(1)[0]
            return f"The most common object in this image is '{most_common_item}' with {freq} detection(s)."

        # Default query summary
        summary_parts = [f"{count} {name}(s)" for name, count in analysis["class_counts"].items()]
        return (
            f"Image analysis summary: Detected {', '.join(summary_parts) if summary_parts else 'no items'}. "
            f"Worker compliance: {sum(1 for w in analysis['worker_profiles'] if w['compliant'])}/{analysis['total_people']} "
            "workers are fully equipped with both helmet and vest."
        )

spatial_reasoning_engine = SpatialReasoningEngine()
