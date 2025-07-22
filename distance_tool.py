"""
Used to answer distance based queries. Uses existing POI database.
WITH DEBUG STATEMENTS
"""

from math import sqrt
import os
import logging
import json
import re
from typing import List, Dict, Optional, Tuple, Type
from langchain.tools import BaseTool
from langchain_community.vectorstores.chroma import Chroma
from pydantic import BaseModel, Field, field_validator
from get_embedding_function import get_embedding_function

# log info - set to DEBUG for more info
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

CHROMA_PATH = "chroma"

class NaturalPOISearchEngine:

    def __init__(self, chroma_path: str = CHROMA_PATH):
        logger.debug("=== Initializing NaturalPOISearchEngine ===")
        self.chroma_path = chroma_path
        self.embedding_function = get_embedding_function()
        self.db = None
        self._initialize_db()

    def _initialize_db(self):
        logger.debug("Attempting to connect to Chroma DB...")
        try:
            self.db = Chroma(
                persist_directory=self.chroma_path,
                embedding_function=self.embedding_function
            )
            logger.info(f"✓ Successfully connected to Chroma database at {self.chroma_path}")
        except Exception as e:
            logger.error(f"✗ Failed to connect to Chroma: {e}")

    def _parse_poi_data(self, doc_content: str) -> Dict:
        """
        Parse POI data from the key-value format stored in Chroma
        """
        logger.debug(f"=== Parsing POI data ===")
        logger.debug(f"Raw content (first 200 chars): {doc_content[:200]}...")
        
        # fields to be stored
        poi_data = {
            'name': None,
            'category': None,
            'coordinates': None,
            'description': None,
            'keywords': None,
            'status': 'active'
        }
        
        try:
            lines = doc_content.strip().split('\n')
            logger.debug(f"Number of lines: {len(lines)}")
            
            # extracts value from key-value pairs
            data_dict = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    data_dict[key.strip()] = value.strip()
            
            logger.debug(f"Parsed fields: {list(data_dict.keys())}")
            
            
            poi_data['name'] = data_dict.get('poi_name')
            poi_data['category'] = data_dict.get('category', '').lower()
            poi_data['keywords'] = data_dict.get('keywords', '')
            poi_data['status'] = data_dict.get('status', 'active').lower()
            poi_data['description'] = data_dict.get('description', '')
            
            # store latitude and longitude as a tuple
            lat_str = data_dict.get('latitude', '').strip()
            lon_str = data_dict.get('longitude', '').strip()
            
            logger.debug(f"Coordinate strings - lat: '{lat_str}', lon: '{lon_str}'")
            
            if lat_str and lon_str:
                try:
                    lat = float(lat_str)
                    lon = float(lon_str)
                    # new tuple
                    poi_data['coordinates'] = (lat, lon)
                    logger.debug(f"Parsed coordinates: {poi_data['coordinates']}")
                    # raise error.
                except ValueError as e:
                    logger.error(f"Failed to parse coordinates: {lat_str}, {lon_str} - {e}")
                    poi_data['coordinates'] = None
            else:
                logger.debug("Missing latitude or longitude")
                    
        except Exception as e:
            logger.error(f"Error parsing POI data: {e}")
        
        logger.debug(f"Final parsed POI: name={poi_data['name']}, coords={poi_data['coordinates']}, status={poi_data['status']}")
        return poi_data
    
    def _enhance_query(self, query: str) -> str:
        logger.debug(f"Enhancing query: '{query}'")
        enhancements = {
            "coffee": "coffee cafe espresso latte cappuccino coffeehouse coffeeshop قهوة كافيه",
            "كافيه": "coffee cafe قهوة كافيه مقهى",
            "قهوة": "coffee قهوة كافيه مقهى",
            
            # Food related
            "restaurant": "restaurant dining food eat مطعم طعام",
            "مطعم": "restaurant مطعم طعام أكل",
            "eat": "restaurant food dining eat طعام أكل",
            "food": "restaurant food dining طعام أكل مطعم",
            
            # Business related
            "bank": "bank banking financial بنك مصرف",
            "بنك": "bank بنك مصرف مالي",
            
            # Shopping
            "shop": "shop shopping retail store متجر تسوق",
            "متجر": "shop متجر محل تسوق",
            
            # Other facilities
            "parking": "parking car vehicle موقف سيارة",
            "موقف": "parking موقف سيارة",
            "prayer": "prayer mosque masjid صلاة مسجد",
            "مسجد": "mosque masjid prayer مسجد صلاة",
        }

        enhanced = query
        query_lower = query.lower()

        # add enhancements
        for key, value in enhancements.items():
            if key in query_lower:
                enhanced += f" {value}"
                logger.debug(f"Added enhancement for '{key}'")
        
        logger.debug(f"Enhanced query: '{enhanced}'")
        return enhanced
    
    def get_poi_by_name(self, name: str) -> Optional[Dict]:
        logger.debug(f"=== Getting POI by name: '{name}' ===")
        
        # log
        if not self.db:
            logger.error("Database not initialized")
            return None
        
        # get 3 most similar pois. user may have typos
        results = self.db.similarity_search(name, k=3)
        logger.debug(f"Found {len(results)} results from similarity search")

        # iterate through returned pois
        for i, doc in enumerate(results):
            poi_data = self._parse_poi_data(doc.page_content)
            poi_name = poi_data.get('name')
            
            logger.debug(f"Result {i}: name='{poi_name}', searching for '{name}'")
            
            # in database, return parsed data
            if poi_name and name.lower() in poi_name.lower():
                logger.info(f"Found match: '{poi_name}'")
                return {
                    'name': poi_name,
                    'content': doc.page_content,
                    'coordinates': poi_data.get('coordinates'),
                    'category': poi_data.get('category'),
                    'description': poi_data.get('description'),
                    'keywords': poi_data.get('keywords'),
                    'status': poi_data.get('status')
                }
        
        logger.warning(f"No match found for '{name}'")
        return None
    
    # from poi1 to poi2
    def calculate_distance(self, poi1_name: str, poi2_name: str) -> Optional[float]:
        logger.debug(f"=== Calculating distance from '{poi1_name}' to '{poi2_name}' ===")
        
        # get names from inputted agent fields
        poi1 = self.get_poi_by_name(poi1_name)
        poi2 = self.get_poi_by_name(poi2_name)

        # none returned by name search
        if not poi1:
            logger.error(f"POI1 '{poi1_name}' not found")
            return None
        if not poi2:
            logger.error(f"POI2 '{poi2_name}' not found")
            return None
        
        # if the poi doesnt have inputted coordinates
        if not poi1.get('coordinates'):
            logger.error(f"POI1 '{poi1_name}' has no coordinates")
            return None
        if not poi2.get('coordinates'):
            logger.error(f"POI2 '{poi2_name}' has no coordinates")
            return None
        
        x1, y1 = poi1['coordinates']
        x2, y2 = poi2['coordinates']
        
        logger.debug(f"Coordinates: POI1=({x1}, {y1}), POI2=({x2}, {y2})")

        # euclidian distance
        distance = sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        logger.info(f"Calculated distance: {distance:.2f} units")
        return distance
    
    def find_nearest(self, reference_poi: str, query: str, limit: int = 3) -> List[Dict]:
        logger.debug(f"=== Finding nearest '{query}' to '{reference_poi}' (limit={limit}) ===")
        
        if not self.db:
            logger.error("Database not initialized")
            return []
            
        # Get reference POI
        ref_poi = self.get_poi_by_name(reference_poi)
        if not ref_poi:
            logger.error(f"Reference POI '{reference_poi}' not found")
            return []
        if not ref_poi.get('coordinates'):
            logger.error(f"Reference POI '{reference_poi}' has no coordinates")
            return []
            
        # Enhance query for better matching
        enhanced_query = self._enhance_query(query)
        
        # Search for matching POIs
        results = self.db.similarity_search(enhanced_query, k=30)
        logger.debug(f"Found {len(results)} potential matches")
        
        # Calculate distances and filter duplicates
        poi_with_distances = []
        seen_names = set()
        
        for doc in results:
            poi_data = self._parse_poi_data(doc.page_content)
            
            # Skip inactive POIs
            if poi_data.get('status') != 'active':
                logger.debug(f"Skipping inactive POI: {poi_data.get('name')}")
                continue
            
            poi_name = poi_data.get('name')
            
            # Skip if we've already seen this POI or if it's the reference
            if not poi_name or poi_name in seen_names or poi_name.lower() == ref_poi['name'].lower():
                continue
            
            if poi_data.get('coordinates'):
                distance = self.calculate_distance(ref_poi['name'], poi_name)
                if distance is not None:
                    poi_with_distances.append({
                        'name': poi_name,
                        'category': poi_data.get('category', 'unknown'),
                        'description': poi_data.get('description', ''),
                        'keywords': poi_data.get('keywords', ''),
                        'distance': distance
                    })
                    seen_names.add(poi_name)
                    logger.debug(f"Added POI '{poi_name}' with distance {distance:.2f}")
        
        # Sort by distance and return top results
        poi_with_distances.sort(key=lambda x: x['distance'])
        result = poi_with_distances[:limit]
        logger.info(f"Returning {len(result)} nearest POIs")
        return result

# Initialize the search engine
logger.info("=== Initializing search engine ===")
search_engine = NaturalPOISearchEngine()

class DistanceInput(BaseModel):
    poi_name_1: str = Field(description="First POI name (origin)")
    poi_name_2: str = Field(description="Second POI name (destination)")

class GetDistanceTool(BaseTool):
    name: str = "get_distance"
    description: str = "Returns distance in meters between two specific named POIs"
    args_schema: Type[BaseModel] = DistanceInput

    def _run(self, poi_name_1: str, poi_name_2: str) -> str:
        logger.info(f"=== GetDistanceTool._run called with '{poi_name_1}' to '{poi_name_2}' ===")
        
        distance = search_engine.calculate_distance(poi_name_1, poi_name_2)

        if distance is None:
            logger.debug("Distance calculation returned None, checking why...")
            poi1 = search_engine.get_poi_by_name(poi_name_1)
            poi2 = search_engine.get_poi_by_name(poi_name_2)
            
            if not poi1 and not poi2:
                response = f"I couldn't find '{poi_name_1}' or '{poi_name_2}' in KAFD. Please check the names and try again."
            elif not poi1:
                response = f"I couldn't find '{poi_name_1}' in KAFD. Please check the name and try again."
            elif not poi2:
                response = f"I couldn't find '{poi_name_2}' in KAFD. Please check the name and try again."
            elif not poi1.get('coordinates') or not poi2.get('coordinates'):
                response = f"I don't have location data for calculating the distance between {poi_name_1} and {poi_name_2}."
            else:
                response = f"I couldn't calculate the distance between {poi_name_1} and {poi_name_2}."
            
            logger.warning(f"✗ Returning error: {response}")
            return response
            
        distance_km = distance / 1000
        walking_time = int(distance/80)

        response = f"The distance from {poi_name_1} to {poi_name_2} is approximately {distance:.0f} meters"

        if distance_km >= 1:
            response += f" ({distance_km:.1f} km)"
        if walking_time > 0:
            response += f", about {walking_time} minute{'s' if walking_time != 1 else ''} walking"

        response += "."
        logger.info(f"✓ Returning success: {response}")
        return response


class NearestSearchInput(BaseModel):
    reference_poi: str = Field(description="The reference location in KAFD")
    query: str = Field(description="What to search for (e.g., 'coffee', 'restaurant', 'ATM')")
    limit: int = Field(default=3, description="Number of results to return")

    @field_validator('limit', mode='before')
    @classmethod
    def validate_limit(cls, v):
        """Convert string to int and ensure it's within reasonable bounds"""
        logger.debug(f"Validating limit value: {v} (type: {type(v)})")
        if isinstance(v, str):
            try:
                v = int(v)
            except ValueError:
                logger.warning(f"Could not convert limit '{v}' to int, using default 3")
                return 3  # default fallback

        if not isinstance(v, int) or v < 1:
            return 3
        if v > 10:
            return 10
        return v

class FindNearestTool(BaseTool):
    name: str = "find_nearest"
    description: str = "Find the nearest places matching a query (coffee, restaurant, etc.) to a reference location."
    args_schema: Type[BaseModel] = NearestSearchInput

    def _run(self, reference_poi: str, query: str, limit: int = 3) -> str:
        logger.info(f"=== FindNearestTool._run called: '{query}' near '{reference_poi}' (limit={limit}) ===")

        results = search_engine.find_nearest(reference_poi, query, limit)

        if not results:
            ref_poi = search_engine.get_poi_by_name(reference_poi)
            if not ref_poi:
                response = f"I couldn't find '{reference_poi}' in KAFD. Please check the name and try again."
            else:
                response = f"I couldn't find any {query} near {reference_poi} in KAFD."
            
            logger.warning(f"✗ No results found: {response}")
            return response
            
        response = f"Here are the nearest {query} locations to {reference_poi}:\n\n"

        for i, poi in enumerate(results, 1):
            walking_time = int(poi['distance'] / 80)

            response += f"{i}. **{poi['name']}**"
            if poi.get('category'):
                response += f" ({poi['category']})"
            response += f"\n   - Distance: {poi['distance']:.0f} meters"

            if walking_time > 0:
                response += f" ({walking_time} min walk)"
            
            response += "\n"
        
        response += "\nWould you like directions to any of these places?"
        
        logger.info(f"✓ Returning {len(results)} results")
        return response
    
class NaturalSearchInput(BaseModel):
    query: str = Field(description="Natural language query about places or distances in KAFD")

class NaturalSearchTool(BaseTool):
    name: str = "search_kafd"
    description: str = "Search for places in KAFD using natural language (handles 'where can I get coffee', 'distance from X to Y', etc.)"
    args_schema: Type[BaseModel] = NaturalSearchInput
    
    def _run(self, query: str) -> str:
        """Process natural language queries about KAFD locations"""
        logger.info(f"=== NaturalSearchTool._run called with query: '{query}' ===")
        
        query_lower = query.lower()
        
        # Pattern 1: Distance queries
        distance_patterns = [
            r"distance (?:from|between) (.+?) (?:to|and) (.+)",
            r"how far (?:is it )?(?:from )?(.+?) to (.+)",
            r"(.+?) to (.+?) distance",
            r"المسافة (?:من|بين) (.+?) (?:إلى|و) (.+)",
        ]
        
        for pattern in distance_patterns:
            match = re.search(pattern, query_lower)
            if match:
                logger.debug(f"Matched distance pattern: {pattern}")
                poi1, poi2 = match.group(1).strip(), match.group(2).strip()
                logger.debug(f"Extracted POIs: '{poi1}' and '{poi2}'")
                return GetDistanceTool()._run(poi1, poi2)
        
        # Pattern 2: Nearest queries with reference point
        nearest_patterns = [
            r"(.+?) (?:near|close to|around|by) (.+)",
            r"nearest (.+?) to (.+)",
            r"closest (.+?) to (.+)",
            r"أقرب (.+?) (?:من|إلى) (.+)",
        ]
        
        for pattern in nearest_patterns:
            match = re.search(pattern, query_lower)
            if match:
                logger.debug(f"Matched nearest pattern: {pattern}")
                what = match.group(1).strip()
                where = match.group(2).strip()
                logger.debug(f"Looking for '{what}' near '{where}'")
                
                # Swap if needed (e.g., "near Starbucks" vs "coffee near Starbucks")
                if search_engine.get_poi_by_name(what):
                    logger.debug(f"Swapping: '{what}' is a POI, so looking for '{where}' near it")
                    what, where = where, what
                
                return FindNearestTool()._run(where, what)
        
        # Pattern 3: General search without specific reference
        logger.debug("No specific pattern matched, doing general search")
        
        if not search_engine.db:
            return "I'm having trouble accessing the location database. Please try again."
            
        enhanced_query = search_engine._enhance_query(query)
        results = search_engine.db.similarity_search(enhanced_query, k=5)
        
        if not results:
            return "I couldn't find any places matching your query in KAFD."
        
        response = "Here's what I found in KAFD:\n\n"
        seen_names = set()
        count = 0
        
        for doc in results:
            poi_data = search_engine._parse_poi_data(doc.page_content)
            
            # Skip inactive POIs
            if poi_data.get('status') != 'active':
                continue
                
            poi_name = poi_data.get('name')
            if poi_name and poi_name not in seen_names:
                count += 1
                response += f"{count}. **{poi_name}**"
                if poi_data.get('category'):
                    response += f" - {poi_data['category']}"
                response += "\n"
                seen_names.add(poi_name)
                
                if count >= 3:
                    break
        
        if count == 0:
            return "I couldn't find any active places matching your query in KAFD."
            
        response += "\nWould you like more information about any of these places?"
        
        logger.info(f"✓ Returning general search results ({count} items)")
        return response

        
if __name__ == "__main__":
    logger.info("=== Running direct tests ===")
    
    # Test the tools
    distance_tool = GetDistanceTool()
    print("\n" + "="*50)
    print("Testing GetDistanceTool:")
    print("="*50)
    result = distance_tool._run("Starbucks", "Kids Home")
    print(f"Result: {result}")
    
    print("\n" + "="*50)
    print("Testing FindNearestTool:")
    print("="*50)
    nearest_tool = FindNearestTool()
    result = nearest_tool._run("Starbucks", "coffee", 3)
    print(f"Result: {result}")