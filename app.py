import os
import random
import json
import io
from PIL import Image
from google import genai
from google.genai import types
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Load environment variables
load_dotenv()

# Pre-create one persistent Gemini client per API key at startup.
# Creating a new client per request causes "client has been closed" errors
# because the underlying HTTP session gets torn down after the first use.
_raw_keys = [
    os.getenv("GEMINI_API_KEY1"),
    os.getenv("GEMINI_API_KEY2"),
    os.getenv("GEMINI_API_KEY3"),
    os.getenv("GEMINI_API_KEY4"),
]
GEMINI_CLIENTS = [genai.Client(api_key=k) for k in _raw_keys if k]

def get_gemini_client():
    """Returns a random pre-created Gemini client."""
    client = random.choice(GEMINI_CLIENTS)
    return client

app = Flask(__name__)

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")

def get_db_collection():
    """Connects to MongoDB and returns the stats collection"""
    if not MONGO_URI:
        return None
    try:
        client = MongoClient(MONGO_URI)
        db = client.get_default_database()
        return db.stats
    except (ConnectionFailure, Exception) as e:
        print(f"MongoDB connection error: {e}")
        return None

def update_counter(key):
    """Updates a counter in MongoDB"""
    collection = get_db_collection()
    if collection is None:
        return 0
    
    try:
        result = collection.find_one_and_update(
            {"_id": "global_stats"},
            {"$inc": {key: 1}},
            upsert=True,
            return_document=True
        )
        return result.get(key, 0)
    except Exception as e:
        print(f"Error updating counter {key}: {e}")
        return 0

def get_counter_value(key):
    """Gets a counter value from MongoDB"""
    collection = get_db_collection()
    if collection is None:
        return 0
        
    try:
        doc = collection.find_one({"_id": "global_stats"})
        return doc.get(key, 0) if doc else 0
    except Exception as e:
        print(f"Error getting counter {key}: {e}")
        return 0

# Legacy wrappers for backward compatibility if needed, using the new generic system
def update_global_counter():
    return update_counter('total_dishes_scanned')

def get_global_counter():
    return get_counter_value('total_dishes_scanned')

@app.route('/')
def index():
    # Increment visit counter
    visit_count = update_counter('total_visits')
    return render_template('index.html', 
                         initial_count=get_global_counter(),
                         visit_count=visit_count)

@app.route('/analyze', methods=['POST'])
def analyze_dish():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    language = request.form.get('language', 'English')
    
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image bytes
        original_data = image_file.read()
        
        # Resize image to save Gemini quota
        try:
            img = Image.open(io.BytesIO(original_data))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail((512, 512))
            output = io.BytesIO()
            img.save(output, format="JPEG", quality=85)
            image_data = output.getvalue()
            mime_type = "image/jpeg"
        except Exception as e:
            print(f"Error resizing image: {e}")
            image_data = original_data
            mime_type = image_file.content_type
            
        # Prepare the prompt
        prompt = f"""
        Analyze this image of a Vietnamese dish and provide the following information in {language}.
        Return the response in this specific JSON structure:
        {{
            "dish_name": "Name of the dish (provide both Vietnamese name and English name if applicable)",
            "calories": "Estimated calories per serving",
            "macronutrients": "Estimated protein, carbs, fats",
            "health_rating": "Rating from 1-10 (number only)",
            "health_advice": "How to make it healthier (1-2 sentences)",
            "history": "Brief cultural history and key ingredients (2-3 sentences)",
            "recipe_steps": ["Step 1...", "Step 2...", "Step 3..."]
        }}
        """

        print(f"Analyzing image with model: gemini-3.1-flash-lite-preview")
        # Call Gemini API
        response = get_gemini_client().models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_data, mime_type=mime_type),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ],
            config=types.GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                top_k=64,
                max_output_tokens=8192,
                response_mime_type="application/json",
            )
        )
        
        # Parse JSON response
        result = json.loads(response.text)
        
        # Update counter only on success
        new_count = update_global_counter()
        
        # Add counter to response to update frontend immediately
        result['global_count'] = new_count
        
        return jsonify(result)

    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/counter', methods=['GET'])
def get_counter():
    return jsonify({
        'count': get_global_counter(),
        'visits': get_counter_value('total_visits')
    })

if __name__ == '__main__':
    app.run()