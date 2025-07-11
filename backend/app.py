from flask import Flask, request, jsonify
import cv2
import numpy as np
from database import get_features, find_similar

app = Flask(__name__)

@app.route('/match', methods=['POST'])
def match_profile():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Extract features (you'll implement this)
    query_features = extract_features(img)
    
    # Find matches from database
    matches = find_similar(query_features, get_features())
    
    return jsonify({'matches': matches})

def extract_features(img):
    """Implement your feature extraction logic here"""
    # Example: Use Hu Moments as in previous example
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest_contour)
    return cv2.HuMoments(moments).flatten()

if __name__ == '__main__':
    app.run(debug=True)