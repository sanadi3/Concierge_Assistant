"""
Debug script to test distance tools directly
"""

import logging
from distance_tool import search_engine, GetDistanceTool, FindNearestTool

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_direct_search():
    """Test the search engine directly"""
    print("\n=== Testing Search Engine Directly ===")
    
    # Test 1: Can we find any POI?
    test_names = ["Starbucks", "starbucks", "KAFD Grand Mosque", "Capital Market Authority", "CMA"]
    
    for name in test_names:
        print(f"\nSearching for: {name}")
        poi = search_engine.get_poi_by_name(name)
        if poi:
            print(f"  Found: {poi['name']}")
            print(f"  Coordinates: {poi.get('coordinates')}")
            print(f"  Status: {poi.get('status')}")
        else:
            print(f"  NOT FOUND")
    
    # Test 2: Check what the raw documents look like
    print("\n=== Sample Raw Documents ===")
    if search_engine.db:
        results = search_engine.db.similarity_search("coffee", k=3)
        for i, doc in enumerate(results):
            print(f"\nDocument {i+1}:")
            print(f"Content preview: {doc.page_content[:200]}...")
            print(f"Metadata: {doc.metadata}")

def test_tools():
    """Test the tools directly"""
    print("\n=== Testing Tools Directly ===")
    
    # Test distance tool
    distance_tool = GetDistanceTool()
    print("\nTesting GetDistanceTool:")
    result = distance_tool._run("Starbucks", "Capital Market Authority")
    print(f"Result: {result}")
    
    # Test nearest tool
    nearest_tool = FindNearestTool()
    print("\nTesting FindNearestTool:")
    result = nearest_tool._run("KAFD Grand Mosque", "coffee", 3)
    print(f"Result: {result}")

def check_csv_format():
    """Check how the CSV data is being loaded"""
    print("\n=== Checking CSV Format ===")
    
    if search_engine.db:
        # Get a sample document
        results = search_engine.db.get(limit=1)
        if results and results['documents']:
            doc_content = results['documents'][0]
            print("Sample document content:")
            print(doc_content)
            print("\nParsing this document:")
            
            # Try parsing
            poi_data = search_engine._parse_poi_data(doc_content)
            print(f"Parsed data: {poi_data}")

if __name__ == "__main__":
    print("Starting Distance Tool Debug...")
    
    # Run all tests
    check_csv_format()
    test_direct_search()
    test_tools()
    
    print("\n=== Debug Complete ===")