import os
import gdown
import zipfile
import joblib

# We will use a placeholder Google Drive URL for the zip file.
# The zip file is expected to contain preprocessor.pkl, pca.pkl, and model.pkl.
GDRIVE_FILE_ID = "11NZtZTTgmWR-XPcAkTqN4rIENa-PE6Gm"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MODELS_ZIP_PATH = os.path.join(MODELS_DIR, "models.zip")

class ModelManager:
    def __init__(self):
        self.preprocessor = None
        self.pca = None
        self.model = None

    def ensure_models_downloaded(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        # Check if models exist
        required_files = ['preprocessor.pkl', 'pca.pkl', 'model.pkl']
        all_exist = all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required_files)
        
        if not all_exist:
            print("Models not found locally. Downloading from Google Drive...")
            try:
                # If we don't have a valid ID yet, we'll gracefully fallback for the demo
                if GDRIVE_FILE_ID.startswith("1PLACEHOLDER"):
                    print("WARNING: Using a placeholder Drive ID. We will simulate model load.")
                    return False
                
                url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
                gdown.download(url, MODELS_ZIP_PATH, quiet=False)
                
                print("Extracting models...")
                with zipfile.ZipFile(MODELS_ZIP_PATH, 'r') as zip_ref:
                    zip_ref.extractall(MODELS_DIR)
                    
                print("Models downloaded and extracted successfully.")
            except Exception as e:
                print(f"Error downloading models: {e}")
                return False
        
        return True

    def load_models(self):
        if not self.ensure_models_downloaded():
            print("Could not download models. The API will run in mock mode or error out.")
            return False
            
        try:
            print("Loading models into memory...")
            self.preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
            self.pca = joblib.load(os.path.join(MODELS_DIR, 'pca.pkl'))
            self.model = joblib.load(os.path.join(MODELS_DIR, 'model.pkl'))
            print("Models loaded successfully.")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def predict(self, input_df):
        if self.model is None or self.preprocessor is None or self.pca is None:
            raise ValueError("Models are not loaded.")
            
        processed = self.preprocessor.transform(input_df)
        pca_transformed = self.pca.transform(processed)
        prediction = self.model.predict(pca_transformed)
        return prediction

# Singleton instance
manager = ModelManager()
