# Big Data Processing Project - Music Clustering and Analysis

This repository contains the final project for **Big Data Processing** course at Binus University.  
The project focuses on analyzing popular music datasets from Spotify, Apple Music, Deezer, and Shazam, and clustering songs based on audio features using **K-Means**.

---

## 👥 Team Members
- Albertus Christian (2702255703)  
- Archi Setio (2702255962)  
- Dominikus Sebastian Ramli (2702329664)  
- Ignatius Abraham Aristio Kusnadi (2702243590)  
- Vincent Virgo (2702250381)  
- Wilbert Bernardi (2702255786)  

Class: **LD01**  
Lecturer: **Fepri Putra Panghurian, S.Kom, M.T.I.**  
Academic Year: **2024/2025**

---

## 📌 Objectives
- Analyze the correlation between audio attributes (BPM, energy, danceability, etc.) and song popularity.  
- Cluster songs into different groups using **K-Means** based on audio features.  
- Compare popularity across different clusters.  
- Explore artist collaboration and its impact on popularity.  

---

## 🗂 Dataset
- **Source:** Popular Spotify Songs Dataset (includes Spotify, Apple Music, Deezer, and Shazam)  
- **Attributes:**  
  - `track_name`, `artist(s)_name`, `released_year`, `released_month`, `released_day`  
  - `streams` (total plays)  
  - Audio features: `bpm`, `danceability_%`, `energy_%`, `valence_%`, `acousticness_%`, `instrumentalness_%`, `liveness_%`, `speechiness_%`  

---

## 🛠 Tools & Libraries
- **Environment:** Google Colab  
- **Libraries:**  
  - `pandas` → data manipulation  
  - `numpy` → numerical processing  
  - `matplotlib`, `seaborn` → data visualization  
  - `sklearn` → K-Means clustering, PCA  
  - `StandardScaler` → feature normalization  

---

## 🔎 Methodology & Code Examples

### 1. Load and Inspect Data
```python
import pandas as pd

# Load dataset
df = pd.read_csv("spotify_songs.csv", encoding="latin1")

# Check structure
print(df.head())
print(df.info())
```
➡️ Here we import the dataset, handle encoding, and quickly inspect the structure to understand columns and data types.

### 2. Data Cleaning
```python
# Check missing values and duplicates
print(df.isnull().sum())
print(df.duplicated().sum())

# Drop irrelevant columns
df = df.drop(columns=["key"])

# Convert streams to numeric
df["streams"] = pd.to_numeric(df["streams"], errors="coerce")
```
➡️ We removed irrelevant columns, handled missing values, and converted streams into numeric format.

### 3. Exploratory Data Analysis (EDA)
```python

import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot to detect outliers
sns.boxplot(x=df["danceability_%"])
plt.show()

# Artist popularity trend
top_artists = df.groupby("artist(s)_name")["streams"].sum().nlargest(5)
top_artists.plot(kind="barh")
plt.show()
```
➡️ Boxplots were used to detect outliers, while bar charts and line charts visualized artist trends and song popularity.

### 4. K-Means Clustering
```python

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Select features
features = ["valence_%", "danceability_%", "energy_%",
            "acousticness_%", "instrumentalness_%",
            "liveness_%", "speechiness_%"]

X = df[features]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(n_clusters=6, random_state=42)
df["Music_Cluster"] = kmeans.fit_predict(X_scaled)
```
➡️ We applied StandardScaler to normalize features, then used K-Means (K=6) for clustering.

### 5. Cluster Visualization
```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
reduced = pca.fit_transform(X_scaled)

plt.scatter(reduced[:,0], reduced[:,1], c=df["Music_Cluster"], cmap="tab10")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("K-Means Clustering of Songs")
plt.show()
```
➡️ PCA reduces dimensions into 2D for visualization. Each color represents a different cluster of songs.

--- 
📊 Results

6 distinct clusters were identified, each representing unique musical styles.

The cluster "Mellow & Reflective Acoustic Pieces" had the highest average streams.

Other popular clusters: Intense & Brooding Rhythmic Tracks, Upbeat & Joyful Dance Hits.

Example representative songs per cluster: Shape of You, Blinding Lights, Someone You Loved.

---

📝 Conclusion

K-Means clustering successfully grouped songs based on audio characteristics.

Popularity varies significantly across clusters, with mellow acoustic tracks attracting the most streams.

This project demonstrates the potential of unsupervised learning in music segmentation and recommendation systems.

---

💡 Future Work

Include more metadata (lyrics, genres) for deeper insights.

Evaluate clustering quality with metrics (e.g., silhouette score).

Experiment with alternative clustering algorithms (e.g., Hierarchical Clustering).

Extend analysis for building playlist recommendation syste
