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

### 2. Data Cleaning

---
