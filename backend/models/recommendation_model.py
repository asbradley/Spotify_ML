import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle
import os

class SpotifyRecommendationModel:
    def __init__(self):
        # Stores KNN model for recommendations
        self.model = None

        # Scaler for normalizing audio features
        self.scaler = None

        # DataFrame to hold the dataset
        self.df = None

        # List of audio features to use
        self.audio_features = [
            'danceability', 'energy', 'key', 'loudness', 'mode', 
            'speechiness', 'acousticness', 'instrumentalness', 
            'liveness', 'valence', 'tempo'
        ]
    
    ''''
    Function to load and clean the dataset
    '''
    def load_data(self, file_path):
        """Load and clean the Spotify dataset"""
        self.df = pd.read_csv(file_path)
        
        # Remove duplicates based on track_name and artists
        print(f"Original dataset size: {len(self.df)} songs")
        self.df = self.df.drop_duplicates(subset=['track_name', 'artists'], keep='first')
        print(f"After removing duplicates: {len(self.df)} songs")
        
        # Reset index after dropping duplicates
        self.df = self.df.reset_index(drop=True)
        
        return self.df
    

    ''''
    Function to preprocess and scale audio features
    '''
    def preprocess_data(self):
        """Preprocess and scale the audio features"""
        X = self.df[self.audio_features].copy()
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
    
    '''
    Function to train KNN model
    Uses cosine similarity to find similar songs
    Updates self.model with trained KNN model
    '''
    def train(self, n_neighbors=10, metric='cosine'):
        """Train the KNN model"""
        X_scaled = self.preprocess_data()
        
        # Initialize and train KNN model
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)

        # Give the mnodel the cleaned data
        self.model.fit(X_scaled)
        
        print(f"Model trained with {len(self.df)} songs")
    
    '''
    Function to get song recommendations
    Searches for given song
    Gets the features from the input song and scales them
    Looops through the nearest neighbors to get recommendations
    Removes duplicates based on track_name and artists
    Returns a list of recommended songs with details
    '''
    def get_recommendations(self, track_name, n_recommendations=5):
        """Get recommendations for a given song"""
        if self.model is None or self.scaler is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Find the song in dataset (case-insensitive search)
        song_matches = self.df[self.df['track_name'].str.lower() == track_name.lower()]
        
        if len(song_matches) == 0:
            return {"error": f"Song '{track_name}' not found in dataset"}
        
        # Use the first match if multiple artists have the same song
        song_idx = song_matches.index[0]
        
        # Get song features and scale them
        song_features = self.df.iloc[song_idx][self.audio_features].values.reshape(1, -1)
        song_features_scaled = self.scaler.transform(song_features)
        
        # Find similar songs (get more than needed to account for potential duplicates)
        distances, indices = self.model.kneighbors(
            song_features_scaled, 
            n_neighbors=min(n_recommendations * 3, len(self.df))  # Get more neighbors
        )
        
        # Format recommendations and remove duplicates
        recommendations = []
        seen_songs = set()
        
        for i in range(1, len(indices[0])):  # Skip the input song itself
            
            # Row index of neighbor in og datafram
            idx = indices[0][i]

            # KNN distance from input song to neighbor
            distance = distances[0][i]
            
            # Create unique identifier for the song
            song_key = (self.df.iloc[idx]['track_name'].lower(), 
                       self.df.iloc[idx]['artists'].lower())
            
            # Skip if we've already seen this song
            if song_key in seen_songs:
                continue
                
            seen_songs.add(song_key)
            
            recommendations.append({
                'track_name': self.df.iloc[idx]['track_name'],
                'artists': self.df.iloc[idx]['artists'],
                'album_name': self.df.iloc[idx]['album_name'],
                'track_genre': self.df.iloc[idx]['track_genre'],
                'popularity': int(self.df.iloc[idx]['popularity']),
                'similarity_score': float(1 - distance)
            })
            
            # Stop when we have enough unique recommendations
            if len(recommendations) >= n_recommendations:
                break
        
        return {
            'input_song': {
                'track_name': self.df.iloc[song_idx]['track_name'],
                'artists': self.df.iloc[song_idx]['artists'],
                'track_genre': self.df.iloc[song_idx]['track_genre'],
                'popularity': int(self.df.iloc[song_idx]['popularity'])
            },
            'recommendations': recommendations
        }
    
    
    def search_songs(self, query, limit=10):
        """Search for songs by name (useful for frontend)"""
        matches = self.df[self.df['track_name'].str.contains(query, case=False, na=False)]
        return matches[['track_name', 'artists', 'track_genre', 'popularity']].head(limit).to_dict('records')
    
    def save_model(self, model_dir='saved_models'):
        """Save the trained model and scaler"""
        os.makedirs(model_dir, exist_ok=True)
        
        with open(f'{model_dir}/knn_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(f'{model_dir}/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
            
        # Save cleaned dataset to data/processed/
        processed_dir = os.path.join(os.path.dirname(model_dir), 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        self.df.to_csv(f'{processed_dir}/cleaned_dataset.csv', index=False)
    
        print(f"Model saved to {model_dir}/")
        print(f"Cleaned dataset saved to {processed_dir}/cleaned_dataset.csv")



    
    def load_model(self, model_dir='saved_models'):
        """Load a pre-trained model"""
        with open(f'{model_dir}/knn_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open(f'{model_dir}/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
            
        # Load from processed data
        processed_dir = os.path.join(os.path.dirname(model_dir), 'data', 'processed')
        self.df = pd.read_csv(f'{processed_dir}/cleaned_dataset.csv')        
        print("Model loaded successfully")

        