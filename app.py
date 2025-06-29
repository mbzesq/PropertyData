import os
import tempfile
from flask import Flask, request, jsonify
from homeharvest import scrape_property
import pandas as pd
import joblib
from pdf2image import convert_from_path
import pytesseract

app = Flask(__name__)

# Load the trained document classification model on startup
MODEL_PATH = "projects/doc_classifier_model.joblib"
doc_model = None
try:
    doc_model = joblib.load(MODEL_PATH)
    print(f"Document classification model '{MODEL_PATH}' loaded successfully.")
except FileNotFoundError:
    print(f"WARNING: Document classification model file not found at '{MODEL_PATH}'. The /predict endpoint will not work.")

@app.route('/')
def health_check():
    """A simple health check endpoint."""
    model_status = "loaded" if doc_model is not None else "not_loaded"
    return jsonify({
        "status": "ok", 
        "message": "PropertyData API is running.",
        "document_classification_model": model_status
    })

@app.route('/predict', methods=['POST'])
def predict_document_type():
    """
    Accepts a PDF file upload and returns the predicted document type for each page.
    """
    if doc_model is None:
        return jsonify({"error": "Document classification model not loaded. Cannot perform prediction."}), 503

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF files are supported"}), 400

    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name

        # Use pdf2image and pytesseract to extract text, just like in the training script
        images = convert_from_path(temp_path)
        
        predictions = []
        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img)
            # The model expects a list of documents, so we pass the text in a list
            prediction = doc_model.predict([text])
            
            # Get prediction confidence/probability if available
            try:
                prediction_proba = doc_model.predict_proba([text])
                confidence = float(max(prediction_proba[0]))
            except AttributeError:
                # Model doesn't support predict_proba
                confidence = None
            
            predictions.append({
                "page": i + 1,
                "predicted_label": prediction[0],
                "confidence": confidence,
                "text_length": len(text)
            })
        
        # Clean up temporary file
        os.unlink(temp_path)
            
        return jsonify({
            "success": True,
            "filename": file.filename,
            "page_count": len(predictions),
            "predictions": predictions
        })

    except Exception as e:
        # Clean up temporary file if it exists
        try:
            os.unlink(temp_path)
        except:
            pass
        
        print(f"An error occurred during document prediction: {e}")
        return jsonify({"error": "Failed to process PDF", "details": str(e)}), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Get information about the loaded document classification model.
    """
    if doc_model is None:
        return jsonify({
            "model_loaded": False,
            "error": "Document classification model not loaded"
        }), 503
    
    try:
        # Try to load model metadata if available
        metadata_path = "projects/doc_classifier_model_metadata.json"
        metadata = {}
        try:
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            pass
        
        # Get model information
        model_info = {
            "model_loaded": True,
            "model_type": str(type(doc_model)),
            "model_path": MODEL_PATH
        }
        
        # Add metadata if available
        if metadata:
            model_info.update({
                "accuracy": metadata.get("accuracy"),
                "training_samples": metadata.get("training_samples"),
                "test_samples": metadata.get("test_samples"),
                "supported_labels": metadata.get("labels"),
                "min_accuracy_threshold": metadata.get("min_accuracy_threshold")
            })
        
        return jsonify(model_info)
        
    except Exception as e:
        return jsonify({
            "model_loaded": True,
            "error": f"Could not retrieve model information: {str(e)}"
        }), 500

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
    Get available search options and parameters for property enrichment.
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

@app.route('/api-info', methods=['GET'])
def api_info():
    """
    Get information about all available API endpoints.
    """
    model_status = "loaded" if doc_model is not None else "not_loaded"
    
    return jsonify({
        "service_name": "PropertyData API",
        "version": "1.0",
        "services": {
            "property_enrichment": {
                "status": "available",
                "endpoints": ["/enrich", "/search-options"],
                "description": "Real estate property data enrichment using Realtor.com scraping"
            },
            "document_classification": {
                "status": "available" if doc_model is not None else "unavailable",
                "model_status": model_status,
                "endpoints": ["/predict", "/model-info"],
                "description": "Document classification for loan collateral analysis"
            }
        },
        "endpoints": {
            "/": "Health check",
            "/enrich": "POST - Enrich property data by address",
            "/search-options": "GET - Get property search options",
            "/predict": "POST - Classify PDF document pages",
            "/model-info": "GET - Get document classification model information",
            "/api-info": "GET - Get API information"
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)