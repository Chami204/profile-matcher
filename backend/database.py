import os
import cv2
import numpy as np
import pickle

DATABASE_FILE = 'features.pkl'

def get_features():
    """Load pre-computed features"""
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'rb') as f:
            return pickle.load(f)
    return {}

def find_similar(query_features, database, top_n=3):
    """Find similar profiles"""
    similarities = []
    for name, db_features in database.items():
        dist = np.linalg.norm(query_features - db_features)
        similarities.append((name, dist))
    
    return sorted(similarities, key=lambda x: x[1])[:top_n]

def build_database(profiles_dir):
    """Process all profile images and extract features"""
    database = {}
    for filename in os.listdir(profiles_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(profiles_dir, filename))
            features = extract_features(img)  # You need to implement this
            database[filename] = features
    
    # Save the database
    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(database, f)
    
    return database