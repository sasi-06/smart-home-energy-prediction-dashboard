import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure the models and plots directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

class ElectricityMLPipeline:
    def __init__(self):
        self.pipeline = None
        self.preprocessor = None
        self.pca = None
        self.model = None
        self.df = None

    # 2.1 Data Collection
    def collect_data(self, filepath):
        print("2.1 Data Collection: Loading dataset...")
        self.df = pd.read_csv(filepath)
        print(f"Dataset loaded with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
        return self.df

    # 2.2 Pre-processing
    def preprocess_data(self):
        print("2.2 Pre-processing: Extracting features and handling dates...")
        df = self.df.copy()
        
        # Drop Home ID as it's just an identifier
        if 'Home ID' in df.columns:
            df = df.drop(columns=['Home ID'])
            
        # Parse Time and Date
        df['Hour'] = pd.to_datetime(df['Time'], format="%H:%M").dt.hour
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        
        # Drop the original string variants of Date and Time
        df = df.drop(columns=['Time', 'Date'])
        
        return df

    # 2.3 Standardisation, 2.4 PCA, and 2.5 Model building are combined into a pipeline
    def build_and_train(self):
        print("Building Model Pipeline (Standardisation -> PCA -> Model)...")
        df_processed = self.preprocess_data()
        
        # Define Features and Target
        X = df_processed.drop(columns=['Energy Consumption (kWh)'])
        y = df_processed['Energy Consumption (kWh)']
        
        # Identify numerical and categorical columns
        numeric_features = ['Outdoor Temperature (°C)', 'Household Size', 'Hour', 'Month', 'DayOfWeek']
        categorical_features = ['Appliance Type', 'Season']
        
        # 2.3 Standardisation & Encoding Preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        
        # We split the data BEFORE PCA to prevent data leakage
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # First Transform data to determine PCA components
        print("Applying Preprocessor...")
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # 2.4 PCA
        # Let's find the number of components for 95% variance
        print("2.4 PCA: Calculating components...")
        self.pca = PCA(n_components=0.95)
        X_train_pca = self.pca.fit_transform(X_train_processed)
        X_test_pca = self.pca.transform(X_test_processed)
        
        num_components = self.pca.n_components_
        print(f"PCA reduced dimensions from {X_train_processed.shape[1]} to {num_components} while retaining 95% variance.")
        
        # 2.5 Model Building
        print("2.5 Model building: Training Random Forest Regressor...")
        self.model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        self.model.fit(X_train_pca, y_train)
        
        # 2.6 Evaluating results
        print("2.6 Evaluating results...")
        y_pred = self.model.predict(X_test_pca)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        print("\n--- Evaluation Metrics ---")
        print(f"Mean Absolute Error (MAE): {mae:.4f} kWh")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        
        # Determine qualitative description for R2 score
        if r2 >= 0.7:
            r2_text = "(This indicates strong predictive capability)"
        elif r2 >= 0.5:
            r2_text = "(This indicates moderate predictive capability)"
        else:
            r2_text = "(This indicates weak predictive capability)"
            
        print(f"R² Score: {r2:.4f} {r2_text}")
        
        # Save components
        print("Saving model artifacts...")
        joblib.dump(self.preprocessor, 'models/preprocessor.pkl')
        joblib.dump(self.pca, 'models/pca.pkl')
        joblib.dump(self.model, 'models/model.pkl')
        
        # 2.8 Visualization
        self.generate_visualizations(y_test, y_pred, df_processed)

    # 2.7 Prediction
    @staticmethod
    def predict_new(data_dict):
        # Load artifacts
        preprocessor = joblib.load('models/preprocessor.pkl')
        pca = joblib.load('models/pca.pkl')
        model = joblib.load('models/model.pkl')
        
        new_df = pd.DataFrame([data_dict])
        
        # Process (assuming the input strictly mimics the df structure)
        # Assuming the caller has already handled Time/Date -> Hour/Month/DayOfWeek
        X_processed = preprocessor.transform(new_df)
        X_pca = pca.transform(X_processed)
        return model.predict(X_pca)[0]

    # 2.8 Visualization
    def generate_visualizations(self, y_test, y_pred, df):
        print("2.8 Visualization: Generating performance plots...")
        sns.set_theme(style="whitegrid")
        
        # 1. Actual vs Predicted Scatter
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, color='#4A90E2')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Energy Consumption (kWh)')
        plt.ylabel('Predicted Energy Consumption (kWh)')
        plt.title('Actual vs Predicted Energy Consumption')
        plt.savefig('plots/actual_vs_predicted.png', bbox_inches='tight')
        plt.show();
        plt.close()
        
        # 2. PCA Explained Variance Ratio
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_), marker='o', linestyle='-', color='#50E3C2')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('PCA Explained Variance')
        plt.grid(True)
        plt.savefig('plots/pca_variance.png', bbox_inches='tight')
        plt.show();
        plt.close()
        
        # 3. Average Consumption by Appliance
        if 'Appliance Type' in df.columns:
            plt.figure(figsize=(12, 6))
            avg_consumption = df.groupby('Appliance Type')['Energy Consumption (kWh)'].mean().sort_values(ascending=False)
            sns.barplot(x=avg_consumption.values, y=avg_consumption.index, palette="viridis")
            plt.xlabel('Average Energy Consumption (kWh)')
            plt.title('Average Energy Consumption by Appliance Type')
            plt.show();
            plt.savefig('plots/appliance_consumption.png', bbox_inches='tight')
            plt.close()

if __name__ == "__main__":
    filepath = "smart_home_energy_consumption_large.csv"
    pipeline = ElectricityMLPipeline()
    pipeline.collect_data(filepath)
    pipeline.build_and_train()
    print("Pipeline execution complete! Models and plots saved.")
