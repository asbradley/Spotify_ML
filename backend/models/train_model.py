import os
from recommendation_model import SpotifyRecommendationModel

def train_and_save_model():
    # Get the backend directory path
    backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(backend_dir, 'data', 'raw', 'dataset.csv')
    
    # Initialize model
    model = SpotifyRecommendationModel()
    
    # Load and train
    print(f"Loading dataset from: {dataset_path}")
    df = model.load_data(dataset_path)
    print(f"Dataset loaded: {len(df)} songs")
    
    print("Training model...")
    model.train(n_neighbors=15, metric='cosine')
    
    # Save the model
    model_dir = os.path.join(backend_dir, 'saved_models')
    model.save_model(model_dir)
    
    print(f"✅ Model training completed!")
    print(f"📁 Saved to: {model_dir}")
    print(f"📊 Songs in model: {len(df)}")

if __name__ == "__main__":
    train_and_save_model()

