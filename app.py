import os
import json
from google import genai
from google.genai import types
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Configure Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

app = Flask(__name__)

# Constants
STATS_FILE = 'stats.json'

def load_stats():
    """Safely loads stats from stats.json"""
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            pass
    return {}

def save_stats(data):
    """Safely saves stats to stats.json"""
    with open(STATS_FILE, 'w') as f:
        json.dump(data, f)

def update_counter(key):
    """Generic function to update any counter in stats.json"""
    data = load_stats()
    current_count = data.get(key, 0)
    current_count += 1
    data[key] = current_count
    save_stats(data)
    return current_count

def get_counter_value(key):
    """Generic function to get any counter value"""
    data = load_stats()
    return data.get(key, 0)

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
        image_data = image_file.read()
        
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

        # Call Gemini API
        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_data, mime_type=image_file.content_type),
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
    # Create stats file if not exists
    if not os.path.exists(STATS_FILE):
        with open(STATS_FILE, 'w') as f:
            json.dump({'total_dishes_scanned': 0}, f)
            
    app.run()
