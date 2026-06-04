# 🎵 Big Data Processing Project - Music Clustering and Analysis

This repository contains the final project for the **Big Data Processing** course at Binus University. The project focuses on analyzing popular music datasets from **Spotify, Apple Music, Deezer, and Shazam**, and clustering songs based on audio features using **K-Means**.

## 👥 Team Members
- Albertus Christian (2702255703)  
- Archi Setio (2702255962)  
- Dominikus Sebastian Ramli (2702329664)  
- Ignatius Abraham Aristio Kusnadi (2702243590)  
- Vincent Virgo (2702250381)  
- Wilbert Bernardi (2702255786)  

**Class:** LD01 | **Lecturer:** Fepri Putra Panghurian, S.Kom, M.T.I. | **Academic Year:** 2024/2025  

## 🎯 My Role & Contributions
In this collaborative project, I took on a dual role as **Technical Co-Lead** and **Project Coordinator**:
- **Data Engineering & Analysis:** Collaborated closely with Dominikus Sebastian Ramli to architect and write the core Python codebase. I handled the end-to-end data pipeline, which included data preprocessing (handling null values and dropping irrelevant columns), conducting Exploratory Data Analysis (generating boxplots and bar charts), and engineering the K-Means clustering model to extract actionable insights.
- **Project Coordination:** Managed project timelines and delegated tasks to ensure all deliverables were completed and submitted punctually. I also orchestrated the team's final presentation strategy to ensure our data insights were communicated clearly and effectively to the audience.

## 📌 Objectives
- Analyze the correlation between audio attributes (BPM, energy, danceability, etc.) and song popularity.  
- Cluster songs into different groups using **K-Means** based on audio features.  
- Compare popularity across different clusters.  
- Explore artist collaboration and its impact on popularity.  

## 🗂 Dataset
- **Source:** Popular Spotify Songs Dataset (includes Spotify, Apple Music, Deezer, and Shazam)  
- **Attributes:**  
  - `track_name`, `artist(s)_name`, `released_year`, `released_month`, `released_day`  
  - `streams` (total plays)  
  - Audio features: `bpm`, `danceability_%`, `energy_%`, `valence_%`, `acousticness_%`, `instrumentalness_%`, `liveness_%`, `speechiness_%`  

## 🛠 Tools & Libraries
- **Environment:** Google Colab  
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn` (K-Means, PCA), `StandardScaler`  

## 🔎 Methodology & Code Examples
1. **Load and Inspect Data**  
   ```python
   import pandas as pd
   df = pd.read_csv("spotify_songs.csv", encoding="latin1")
   print(df.head())
   print(df.info())
   ```
   ➡️ Imported dataset, handled encoding, and inspected structure.  

2. **Data Cleaning**  
   ```python
   print(df.isnull().sum())
   print(df.duplicated().sum())
   df = df.drop(columns=["key"])
   df["streams"] = pd.to_numeric(df["streams"], errors="coerce")
   ```
   ➡️ Removed irrelevant columns, handled missing values, and converted streams to numeric.  

3. **Exploratory Data Analysis (EDA)**  
   ```python
   import matplotlib.pyplot as plt
   import seaborn as sns
   sns.boxplot(x=df["danceability_%"]); plt.show()
   top_artists = df.groupby("artist(s)_name")["streams"].sum().nlargest(5)
   top_artists.plot(kind="barh"); plt.show()
   ```
   ➡️ Used boxplots to detect outliers and bar charts for artist popularity trends.  

4. **K-Means Clustering**  
   ```python
   from sklearn.preprocessing import StandardScaler
   from sklearn.cluster import KMeans
   features = ["valence_%","danceability_%","energy_%","acousticness_%","instrumentalness_%","liveness_%","speechiness_%"]
   X_scaled = StandardScaler().fit_transform(df[features])
   kmeans = KMeans(n_clusters=6, random_state=42)
   df["Music_Cluster"] = kmeans.fit_predict(X_scaled)
   ```
   ➡️ Normalized features and applied K-Means with 6 clusters.  

5. **Cluster Visualization**  
   ```python
   from sklearn.decomposition import PCA
   reduced = PCA(n_components=2).fit_transform(X_scaled)
   plt.scatter(reduced[:,0], reduced[:,1], c=df["Music_Cluster"], cmap="tab10")
   plt.xlabel("PCA 1"); plt.ylabel("PCA 2"); plt.title("K-Means Clustering of Songs"); plt.show()
   ```
   ➡️ PCA reduced dimensions into 2D for visualization. Each color represents a cluster.  

## 📊 Results
- Identified **6 distinct clusters**, each representing unique musical styles.  
- The cluster **"Mellow & Reflective Acoustic Pieces"** had the highest average streams.  
- Other popular clusters included *Intense & Brooding Rhythmic Tracks* and *Upbeat & Joyful Dance Hits*.  
- **Representative songs per cluster:** *Shape of You*, *Blinding Lights*, *Someone You Loved*.  

## 📝 Conclusion
K-Means clustering effectively grouped songs based on audio features. Popularity varied significantly across clusters, with mellow acoustic tracks performing best. This project demonstrates the potential of **unsupervised learning** in music segmentation and recommendation systems.  

## 💻 Link to Google Colab
👉 [Open Colab Notebook](https://colab.research.google.com/drive/1ayoA9CKOmdZS9Crw2avhBxULQjzMXWCv?usp=drive_link)  

## 💡 Future Work
- Incorporate additional metadata (**lyrics, genres**) for deeper insights.  
- Evaluate clustering quality with metrics (e.g., **silhouette score**).  
- Experiment with alternative clustering methods (e.g., **Hierarchical Clustering**).  
- Extend analysis to build a **playlist recommendation system**.  
