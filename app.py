import os
from flask import Flask, request, jsonify
from homeharvest import scrape_property
import pandas as pd

app = Flask(__name__)

@app.route('/')
def health_check():
    """A simple health check endpoint."""
    return jsonify({"status": "ok", "message": "PropertyData API is running."})

@app.route('/enrich', methods=['POST'])
def enrich_property():
    """
    The main endpoint to enrich a property address.
    Expects a JSON body with an "address" key.
    Optional parameters:
    - listing_type: "for_sale", "for_rent", "sold", "pending" (default: "for_sale")
    - past_days: number of days to look back for sold properties
    - radius: search radius in miles (for individual addresses)
    - mls_only: boolean to only fetch MLS listings
    - limit: maximum number of results (default: 10000, max: 10000)
    """
    if not request.json or 'address' not in request.json:
        return jsonify({"error": "Missing 'address' in request body"}), 400

    address = request.json['address']
    listing_type = request.json.get('listing_type', 'for_sale')
    past_days = request.json.get('past_days', None)
    radius = request.json.get('radius', None)
    mls_only = request.json.get('mls_only', False)
    limit = request.json.get('limit', 100)  # Default to 100 for API performance
    
    try:
        # Call the homeharvest scrape_property function
        properties = scrape_property(
            location=address,
            listing_type=listing_type,
            past_days=past_days,
            radius=radius,
            mls_only=mls_only,
            limit=limit
        )
        
        # Check if we got any results
        if isinstance(properties, pd.DataFrame) and not properties.empty:
            # Convert DataFrame to dictionary for JSON response
            result = properties.to_dict(orient='records')
            
            # Clean up NaN values for JSON serialization
            for record in result:
                for key, value in record.items():
                    if pd.isna(value):
                        record[key] = None
            
            return jsonify({
                "success": True,
                "count": len(result),
                "properties": result
            })
        else:
            return jsonify({
                "success": True,
                "count": 0,
                "properties": [],
                "message": "No properties found for the given address"
            })
            
    except ValueError as e:
        # Handle validation errors
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        # Log the exception for debugging
        print(f"An error occurred during enrichment: {e}")
        return jsonify({"error": "An internal error occurred", "details": str(e)}), 500

@app.route('/search-options', methods=['GET'])
def search_options():
    """
    Get available search options and parameters.
    """
    return jsonify({
        "listing_types": ["for_sale", "for_rent", "sold", "pending"],
        "property_types": [
            "single_family", "multi_family", "condos", 
            "condo_townhome_rowhome_coop", "condo_townhome", 
            "townhomes", "duplex_triplex", "farm", "land", "mobile"
        ],
        "parameters": {
            "address": "Required. Location to search (e.g. 'Dallas, TX', '85281', '2530 Al Lipscomb Way')",
            "listing_type": "Optional. Type of listing (default: 'for_sale')",
            "past_days": "Optional. Get properties sold/listed in the last N days",
            "radius": "Optional. Search radius in miles (for individual addresses)",
            "mls_only": "Optional. Boolean to fetch only MLS listings",
            "limit": "Optional. Maximum results to return (default: 100, max: 10000)"
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)