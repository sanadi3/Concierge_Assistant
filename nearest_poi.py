import json
from typing import Dict, Tuple, List
from scipy.spatial import KDTree



POI_COORDS: Dict[str, Tuple[float, float]] = {
    "Black Tap": (100.0, 500.0),
    "Starbucks": (110.0, 520.0),
    "12 Cups": (90.0, 480.0),
    "Jones the Grocer": (150.0, 510.0),
    "Banoon Day Care": (300.0, 600.0),
    "KAFD Mosque": (130.0, 470.0),
}

CATEGORIES: Dict[str, List[str]] = {
    "coffee shop": ["Starbucks", "12 Cups", "Jones the Grocer"],
    "day_care":    ["Banoon Day Care"],
    "restaurant":  ["Black Tap", "Jones the Grocer"],
    "mosque":      ["KAFD Mosque"],
}

def find_nearest_poi(origin: str, category: str) -> str:
    """
    Given an origin POI and a target category 
    return the nearest POI in that category
    """

    if origin not in POI_COORDS:
        return f"'{origin}' is not valid."

    candidates = CATEGORIES.get(category.lower())
    if not candidates:
        return f"no pois in category '{category}'."
    
    origin_coords = POI_COORDS[origin]
    candidate_coords = [POI_COORDS[poi] for poi in candidates]

    tree = KDTree(candidate_coords)
    dist, idx = tree.query(origin_coords)

    nearest = candidates[idx]
    distance = int(dist)

    return (
        f"The closest {category} to {origin} is **{nearest}, "
        f"{distance} meters away."
    )

if __name__ == "__main__":
    print("Nearest POI Finder (KDTree)\n")

    while True:
        origin = input("From which POI ").strip()
        category = input("What category? ").strip()

        result = find_nearest_poi(origin, category)
        print(result, "\n")

