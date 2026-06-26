"""
SmartVisionX Location Manager
=============================

Persists and matches user-saved locations. Each reference image has a
perceptual hash + colour histogram. Optionally a "path_signature" is
stored too (used by the arrival fusion module).

M4 additions:
  - Each reference may also store a "path_profile" — a 10-metric
    statistical descriptor of the path environment at save time.
  - Helpers build_path_profile() and get_path_profile() are added.

JSON layout:  Locations/locations.json
Reference images:  Locations/references/<name>_<ms>.jpg
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from Pipelines.path_metrics import PathProfile, compute_metrics  # M4


BASE_DIR = Path(__file__).resolve().parent.parent
LOCATIONS_DIR = BASE_DIR / "Locations"
REFERENCES_DIR = LOCATIONS_DIR / "references"
LOCATIONS_FILE = LOCATIONS_DIR / "locations.json"


_HASH_SIZE = 8


def compute_phash(image: np.ndarray, hash_size: int = _HASH_SIZE) -> str:
    if image is None or image.size == 0:
        return "0" * (hash_size * hash_size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4),
                         interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(resized))
    low = dct[:hash_size, :hash_size]
    med = float(np.median(low))
    bits = (low > med).flatten()
    return "".join("1" if b else "0" for b in bits)


def hamming_distance(hash1: str, hash2: str) -> int:
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return max(len(hash1 or ""), len(hash2 or ""))
    return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))


def compute_color_histogram(image: np.ndarray) -> List[float]:
    if image is None or image.size == 0:
        return [0.0] * 64
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.astype(float).tolist()


def histogram_similarity(h1: List[float], h2: List[float]) -> float:
    if not h1 or not h2 or len(h1) != len(h2):
        return 0.0
    a = np.clip(np.asarray(h1, dtype=np.float32), 0, None)
    b = np.clip(np.asarray(h2, dtype=np.float32), 0, None)
    return float(max(-1.0, min(1.0, cv2.compareHist(a, b, cv2.HISTCMP_CORREL))))


class LocationManager:
    HASH_WEIGHT = 0.60
    HIST_WEIGHT = 0.40
    BEST_WEIGHT = 0.85
    AVG_WEIGHT = 0.15

    def __init__(self, locations_file: Optional[Path] = None) -> None:
        self.locations_file = Path(locations_file) if locations_file else LOCATIONS_FILE
        self.references_dir = REFERENCES_DIR
        self.locations_file.parent.mkdir(parents=True, exist_ok=True)
        self.references_dir.mkdir(parents=True, exist_ok=True)
        self.locations: Dict[str, Dict[str, Any]] = self._load()

    def _load(self) -> Dict[str, Dict[str, Any]]:
        if not self.locations_file.exists():
            return {}
        try:
            with open(self.locations_file, "r", encoding="utf-8") as f:
                return json.load(f) or {}
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[Locations] Failed to load: {exc}")
            return {}

    def save(self) -> None:
        with open(self.locations_file, "w", encoding="utf-8") as f:
            json.dump(self.locations, f, indent=2, ensure_ascii=False)

    def reload(self) -> int:
        self.locations = self._load()
        return len(self.locations)

    def add_location(
        self,
        name: str,
        image: np.ndarray,
        *,
        objects: Optional[List[str]] = None,
        scene: Optional[str] = None,
        scene_conf: Optional[float] = None,
        description: str = "",
        path_signature: Optional[List[float]] = None,
        path_profile: Optional[Dict[str, Any]] = None,   # M4
    ) -> Dict[str, Any]:
        if image is None or image.size == 0:
            raise ValueError("Cannot save empty image")
        key = self._normalise_name(name)
        if not key:
            raise ValueError("Location name cannot be empty")

        timestamp = time.time()
        safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
        ref_path = self.references_dir / f"{safe_key}_{int(timestamp * 1000)}.jpg"
        cv2.imwrite(str(ref_path), image)

        reference: Dict[str, Any] = {
            "timestamp":       timestamp,
            "image_path":      str(ref_path.relative_to(BASE_DIR)),
            "phash":           compute_phash(image),
            "color_histogram": compute_color_histogram(image),
        }
        if scene:
            reference["scene"] = scene
        if scene_conf is not None:
            reference["scene_conf"] = float(scene_conf)
        if objects:
            reference["objects"] = [str(o).lower() for o in objects]
        if path_signature:
            reference["path_signature"] = list(path_signature)
        if path_profile:                                       # M4
            reference["path_profile"] = path_profile

        if key in self.locations:
            loc = self.locations[key]
            loc["references"].append(reference)
            loc["updated_at"] = timestamp
            if objects:
                merged = set(loc.get("expected_objects") or [])
                merged.update(o.lower() for o in objects)
                loc["expected_objects"] = sorted(merged)
            if scene and not loc.get("expected_scene"):
                loc["expected_scene"] = scene
            if description:
                loc["description"] = description
        else:
            self.locations[key] = {
                "name":              key,
                "display_name":      " ".join(p.capitalize() for p in key.split("_")),
                "description":       description,
                "created_at":        timestamp,
                "updated_at":        timestamp,
                "match_count":       0,
                "last_matched_at":   None,
                "expected_scene":    scene,
                "expected_objects":  [o.lower() for o in (objects or [])],
                "references":        [reference],
            }
        self.save()
        return self.locations[key]

    def remove_location(self, name: str) -> bool:
        key = self._normalise_name(name)
        if key not in self.locations:
            return False
        loc = self.locations.pop(key)
        for ref in loc.get("references", []):
            try:
                (BASE_DIR / ref["image_path"]).unlink(missing_ok=True)
            except OSError:
                pass
        self.save()
        return True

    def get_location(self, name: str) -> Optional[Dict[str, Any]]:
        if not name:
            return None
        key = self._normalise_name(name)
        if key in self.locations:
            return self.locations[key]
        for k in self.locations:
            if key in k or k in key:
                return self.locations[k]
        return None

    def list_locations(self) -> List[Dict[str, Any]]:
        items = []
        for loc in self.locations.values():
            items.append({
                "name":             loc["name"],
                "display_name":     loc.get("display_name") or loc["name"],
                "description":      loc.get("description", ""),
                "reference_count":  len(loc.get("references", [])),
                "match_count":      loc.get("match_count", 0),
                "updated_at":       loc.get("updated_at"),
                "expected_scene":   loc.get("expected_scene"),
                "expected_objects": loc.get("expected_objects", []),
                "has_path_signature": any(
                    r.get("path_signature") for r in loc.get("references", [])),
                "has_path_profile": any(                      # M4
                    r.get("path_profile") for r in loc.get("references", [])),
            })
        items.sort(key=lambda x: x["name"])
        return items

    def location_names(self) -> List[str]:
        return sorted(self.locations.keys())

    def match_location(self, image, *, threshold=0.0, only=None):
        if image is None or image.size == 0 or not self.locations:
            return None
        if only is not None:
            target = self.get_location(only)
            if target is None:
                return None
            candidates = {target["name"]: target}
        else:
            candidates = self.locations

        phash = compute_phash(image)
        hist = compute_color_histogram(image)

        best, best_score = None, -1.0
        for name, loc in candidates.items():
            score, hash_sim, hist_sim = self._score_against_location(loc, phash, hist)
            if score > best_score:
                best_score = score
                best = {
                    "name":             name,
                    "display_name":     loc.get("display_name") or name,
                    "match_score":      round(score, 3),
                    "hash_sim":         round(hash_sim, 3),
                    "hist_sim":         round(hist_sim, 3),
                    "reference_count":  len(loc.get("references", [])),
                    "expected_scene":   loc.get("expected_scene"),
                    "expected_objects": loc.get("expected_objects", []),
                    "location":         loc,
                }
        if best is None or best_score < threshold:
            return None
        loc = self.locations[best["name"]]
        loc["match_count"] = loc.get("match_count", 0) + 1
        loc["last_matched_at"] = time.time()
        return best

    def get_all_scores(self, image):
        if image is None or image.size == 0 or not self.locations:
            return []
        phash = compute_phash(image)
        hist = compute_color_histogram(image)
        out = []
        for name, loc in self.locations.items():
            score, hash_sim, hist_sim = self._score_against_location(loc, phash, hist)
            out.append({
                "name":         name,
                "display_name": loc.get("display_name") or name,
                "score":        round(score, 3),
                "hash_sim":     round(hash_sim, 3),
                "hist_sim":     round(hist_sim, 3),
            })
        out.sort(key=lambda r: r["score"], reverse=True)
        return out

    # ------------------------------------------------------------------
    # M4 helpers
    # ------------------------------------------------------------------

    def build_path_profile(self, frames_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Build a serialisable path profile from a list of frame-level metric dicts."""
        return PathProfile(frames_metrics).to_dict()

    def get_path_profile(self, name: str) -> Optional[PathProfile]:
        """Load the path profile for a saved location, or None."""
        loc = self.get_location(name)
        if not loc:
            return None
        refs = loc.get("references", [])
        for ref in refs:
            if ref.get("path_profile"):
                return PathProfile.from_dict(ref["path_profile"])
        return None

    @staticmethod
    def _normalise_name(name: str) -> str:
        return (name or "").strip().lower().replace(" ", "_")

    def _score_against_location(self, loc, phash, hist):
        refs = loc.get("references", [])
        if not refs:
            return 0.0, 0.0, 0.0
        per_ref: List[float] = []
        best_hash, best_hist = 0.0, 0.0
        for ref in refs:
            ref_hash = ref.get("phash", "")
            ref_hist = ref.get("color_histogram", [])
            if not ref_hash or not ref_hist:
                continue
            dist = hamming_distance(phash, ref_hash)
            hash_sim = 1.0 - (dist / max(1, len(ref_hash)))
            hist_sim = histogram_similarity(hist, ref_hist)
            per_ref.append(self.HASH_WEIGHT * hash_sim + self.HIST_WEIGHT * hist_sim)
            best_hash = max(best_hash, hash_sim)
            best_hist = max(best_hist, hist_sim)
        if not per_ref:
            return 0.0, 0.0, 0.0
        best = max(per_ref)
        avg = sum(per_ref) / len(per_ref)
        return self.BEST_WEIGHT * best + self.AVG_WEIGHT * avg, best_hash, best_hist
