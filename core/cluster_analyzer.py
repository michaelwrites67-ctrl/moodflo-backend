"""
Cluster Analysis Module
Groups emotion patterns using K-means clustering
"""
import numpy as np
from typing import Dict, List
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ClusterAnalyzer:
    """Analyze emotion patterns using clustering"""
    
    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
    
    def analyze(
        self,
        emotion_series: List[Dict[str, float]],
        energy_timeline: List[float]
    ) -> Dict:
        """
        Perform clustering analysis on emotion patterns
        Returns cluster data with labels and coordinates
        """
        # Create feature matrix
        features = []
        
        for emotion, energy in zip(emotion_series, energy_timeline):
            feature_vector = [
                emotion.get('neutral', 0),
                emotion.get('happy', 0),
                emotion.get('sad', 0),
                emotion.get('angry', 0),
                emotion.get('fearful', 0),
                energy / 100  # Normalize energy to 0-1
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(features_scaled)
        
        # Get cluster centers for visualization
        # Use PCA to reduce to 2D for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(features_scaled)
        
        # Interpret clusters
        cluster_description = self._interpret_clusters(
            kmeans.cluster_centers_,
            scaler
        )
        
        return {
            'n_clusters': self.n_clusters,
            'labels': labels.tolist(),
            'coordinates': coords_2d.tolist(),
            'description': cluster_description
        }
    
    def _interpret_clusters(
        self,
        cluster_centers: np.ndarray,
        scaler: StandardScaler
    ) -> str:
        """
        Interpret cluster centers to provide meaningful descriptions
        """
        # Inverse transform to get original scale
        centers_original = scaler.inverse_transform(cluster_centers)
        
        descriptions = []
        
        for i, center in enumerate(centers_original):
            neutral, happy, sad, angry, fearful, energy = center
            energy_scaled = energy * 100
            
            # Determine cluster characteristics
            if energy_scaled > 60:
                energy_desc = "High Energy"
            elif energy_scaled < 30:
                energy_desc = "Low Energy"
            else:
                energy_desc = "Moderate Energy"
            
            # Dominant emotion
            emotions = {
                'neutral': neutral,
                'happy': happy,
                'sad': sad,
                'angry': angry,
                'fearful': fearful
            }
            dominant = max(emotions, key=emotions.get)
            
            descriptions.append(
                f"Cluster {i}: {energy_desc}, Primarily {dominant.capitalize()}"
            )
        
        return " | ".join(descriptions)
