# Interview Preparation: The Battle of Neighborhoods (Coursera Capstone)

## 1. Project Overview

**Problem Statement:** Analyze and cluster neighborhoods in a city (e.g., London) based on venue data from Foursquare API to help stakeholders make informed decisions about where to live, invest, or open businesses.

**Objective:** Apply data science and machine learning techniques to:
1. Collect geospatial data using Foursquare API
2. Analyze neighborhood characteristics (restaurants, shops, parks)
3. Cluster similar neighborhoods using K-Means
4. Visualize results on interactive maps
5. Provide recommendations based on clusters

**Use Cases:**
- Real estate investment decisions
- Business location planning
- Residential relocation guidance
- Urban planning insights

---

## 2. Technical Concepts

### Geospatial Data Analysis
- **Geocoding:** Convert addresses to coordinates (lat, long)
- **Reverse Geocoding:** Coordinates to addresses
- **Distance Calculation:** Haversine formula
- **Mapping:** Folium for interactive maps

### Clustering
- **K-Means:** Unsupervised clustering algorithm
- **Elbow Method:** Choose optimal K
- **Silhouette Score:** Cluster quality metric

### APIs
- **Foursquare API:** Venue data (restaurants, cafes, gyms, etc.)
- **Geopy:** Geocoding services
- **Folium:** Interactive mapping

---

## 3. Mathematical Foundations

### Haversine Distance (Great Circle Distance)
\[
a = \sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1) \times \cos(\phi_2) \times \sin^2\left(\frac{\Delta\lambda}{2}\right)
\]
\[
c = 2 \times \text{atan2}(\sqrt{a}, \sqrt{1-a})
\]
\[
d = R \times c
\]
where \(R\) is Earth's radius (6,371 km), \(\phi\) is latitude, \(\lambda\) is longitude.

### K-Means Clustering
**Objective:**
\[
\min \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
\]

**Algorithm:**
1. Initialize k centroids randomly
2. Assign each point to nearest centroid
3. Update centroids (mean of assigned points)
4. Repeat until convergence

### Elbow Method
Plot within-cluster sum of squares (WCSS) vs K:
\[
\text{WCSS} = \sum_{i=1}^{k} \sum_{x \in C_i} ||x - \mu_i||^2
\]

### Silhouette Score
\[
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
\]
where:
- \(a(i)\): Average distance to points in same cluster
- \(b(i)\): Average distance to points in nearest cluster

Range: [-1, 1], higher is better.

---

## 4. Implementation Details

### Complete Workflow

**1. Data Collection**
```python
import pandas as pd
import numpy as np
import requests
from geopy.geocoders import Nominatim

# Load neighborhood data
df = pd.read_csv('london_neighborhoods.csv')

# Geocode neighborhoods
geolocator = Nominatim(user_agent="neighborhood_analysis")

def get_coordinates(neighborhood):
    location = geolocator.geocode(f"{neighborhood}, London, UK")
    if location:
        return location.latitude, location.longitude
    return None, None

df[['latitude', 'longitude']] = df['neighborhood'].apply(
    lambda x: pd.Series(get_coordinates(x))
)
```

**2. Foursquare API Integration**
```python
import requests

def get_venues(lat, lng, radius=500, limit=100):
    """
    Get venues near coordinates using Foursquare API.
    
    Args:
        lat: Latitude
        lng: Longitude
        radius: Search radius (meters)
        limit: Maximum venues to return
    """
    url = 'https://api.foursquare.com/v3/places/search'
    
    headers = {
        'Accept': 'application/json',
        'Authorization': 'YOUR_API_KEY'
    }
    
    params = {
        'll': f'{lat},{lng}',
        'radius': radius,
        'limit': limit
    }
    
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    
    venues = []
    for venue in data.get('results', []):
        venues.append({
            'name': venue['name'],
            'category': venue['categories'][0]['name'] if venue.get('categories') else 'Unknown',
            'lat': venue['geocodes']['main']['latitude'],
            'lng': venue['geocodes']['main']['longitude']
        })
    
    return pd.DataFrame(venues)

# Get venues for each neighborhood
all_venues = []
for idx, row in df.iterrows():
    venues = get_venues(row['latitude'], row['longitude'])
    venues['neighborhood'] = row['neighborhood']
    all_venues.append(venues)

venues_df = pd.concat(all_venues, ignore_index=True)
```

**3. Feature Engineering (One-Hot Encoding)**
```python
# Create venue category columns
venues_onehot = pd.get_dummies(venues_df['category'], prefix='')

# Aggregate by neighborhood
neighborhood_features = pd.DataFrame(
    venues_onehot.groupby(venues_df['neighborhood']).mean()
)

print(neighborhood_features.shape)  # (neighborhoods, venue_categories)
```

**4. K-Means Clustering**
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Scale features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(neighborhood_features)

# Elbow method to choose K
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow curve
plt.plot(K_range, wcss, 'bo-')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Choose optimal K (e.g., K=5)
optimal_k = 5
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# Add cluster labels
neighborhood_features['cluster'] = clusters
```

**5. Cluster Analysis**
```python
# Analyze each cluster
for cluster_id in range(optimal_k):
    print(f"\n=== Cluster {cluster_id} ===")
    
    # Get neighborhoods in this cluster
    cluster_neighborhoods = neighborhood_features[
        neighborhood_features['cluster'] == cluster_id
    ].index.tolist()
    
    print(f"Neighborhoods: {', '.join(cluster_neighborhoods)}")
    
    # Top venue categories
    cluster_data = neighborhood_features[neighborhood_features['cluster'] == cluster_id]
    top_categories = cluster_data.drop('cluster', axis=1).mean().sort_values(ascending=False).head(5)
    
    print("Top Venue Categories:")
    for category, freq in top_categories.items():
        print(f"  {category}: {freq:.3f}")
```

**6. Interactive Mapping**
```python
import folium

# Create base map
map_clusters = folium.Map(
    location=[df['latitude'].mean(), df['longitude'].mean()],
    zoom_start=11
)

# Define colors for clusters
colors = ['red', 'blue', 'green', 'purple', 'orange']

# Add markers
for idx, row in df.iterrows():
    cluster = neighborhood_features.loc[row['neighborhood'], 'cluster']
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=8,
        popup=f"{row['neighborhood']} (Cluster {cluster})",
        color=colors[cluster],
        fill=True,
        fillColor=colors[cluster]
    ).add_to(map_clusters)

# Save map
map_clusters.save('neighborhood_clusters.html')
```

---

## 5. Outcomes & Results

### Typical Findings

**Example Clusters (London):**

**Cluster 0: Business Districts**
- Top venues: Coffee shops, offices, restaurants
- Examples: City of London, Canary Wharf

**Cluster 1: Residential Family Areas**
- Top venues: Parks, schools, supermarkets
- Examples: Suburbs, family neighborhoods

**Cluster 2: Entertainment & Nightlife**
- Top venues: Bars, clubs, theaters
- Examples: Soho, Shoreditch

**Cluster 3: Tourist Areas**
- Top venues: Hotels, museums, landmarks
- Examples: Westminster, Kensington

**Cluster 4: Shopping Districts**
- Top venues: Retail stores, malls, boutiques
- Examples: Oxford Street, Knightsbridge

---

## 6. Interview Questions & Answers

**Q1: How do you choose the optimal number of clusters (K)?**

**A1:**

**Methods:**

**1. Elbow Method:**
- Plot WCSS vs K
- Look for "elbow" (diminishing returns)
```
K=1: High WCSS
K=2: Lower
K=5: Elbow (good choice)
K=10: Minimal improvement
```

**2. Silhouette Score:**
```python
from sklearn.metrics import silhouette_score

silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

optimal_k = np.argmax(silhouette_scores) + 2
```

**3. Domain Knowledge:**
- City typically has 5-7 neighborhood types
- Too few: Overly broad
- Too many: Not useful for decision-making

**Q2: What is the curse of dimensionality in K-Means?**

**A2:**

**Problem:** With many features (venue categories), distances become meaningless.

**Example:**
```
10 dimensions: Points well-separated
1000 dimensions: All points roughly same distance apart!
```

**Solutions:**
```python
# 1. Dimensionality reduction (PCA)
from sklearn.decomposition import PCA
pca = PCA(n_components=20)
features_reduced = pca.fit_transform(features)

# 2. Feature selection
# Keep only most common venue categories
top_categories = features.sum().sort_values(ascending=False).head(50)
features_selected = features[top_categories.index]

# 3. Use different distance metric
# Manhattan distance instead of Euclidean
```

**Q3: How would you validate Foursquare API results?**

**A3:**

**Validation Strategies:**

**1. Cross-Reference:**
```python
# Compare with Google Places API
# Check consistency
```

**2. Manual Inspection:**
```python
# Sample random neighborhoods
# Verify venue categories match reality
```

**3. Statistical Checks:**
```python
# Check for anomalies
print(venues_df['category'].value_counts())
# Ensure reasonable distribution
```

**4. Temporal Validation:**
```python
# Collect data at different times
# Check consistency (venues shouldn't change much)
```

**Q4: How would you present findings to real estate investors?**

**A4:**

**Executive Summary:**

"We analyzed 50 London neighborhoods using geospatial data and identified 5 distinct neighborhood types:

**Cluster 0: Prime Business (Investment Grade A)**
- Characteristics: High office density, premium restaurants
- Recommendation: Commercial real estate, high rental yields
- Examples: City of London, Canary Wharf

**Cluster 1: Family Residential (Investment Grade B)**
- Characteristics: Schools, parks, supermarkets
- Recommendation: Long-term residential, stable tenants
- Examples: Hampstead, Richmond

**Cluster 2: Entertainment Hub (Investment Grade B+)**
- Characteristics: Bars, restaurants, nightlife
- Recommendation: Short-term rentals, hospitality
- Examples: Soho, Shoreditch

**Cluster 3: Tourist Central (Investment Grade A+)**
- Characteristics: Hotels, museums, attractions
- Recommendation: Airbnb, boutique hotels
- Examples: Westminster, Covent Garden

**Cluster 4: Shopping District (Investment Grade A)**
- Characteristics: Retail, luxury brands
- Recommendation: Retail space, mixed-use developments
- Examples: Oxford Street, Knightsbridge

**Visualization:** Interactive map showing clusters, average property prices, rental yields."

---

## Additional Resources

**APIs:**
- Foursquare Places API
- Google Places API
- Mapbox Geocoding

**Tools:**
- Folium: Interactive maps in Python
- GeoPandas: Geospatial data analysis
- Plotly: Interactive visualizations

**Datasets:**
- OpenStreetMap: Geographic data
- Census Data: Demographics by neighborhood

