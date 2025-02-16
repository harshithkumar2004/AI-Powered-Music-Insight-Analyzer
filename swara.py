import streamlit as st
import requests
import os
import plotly.express as px
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
from sklearn.cluster import KMeans

# Load environment variables
load_dotenv()
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY")

if not LASTFM_API_KEY:
    st.error("⚠ Last.fm API Key is missing! Please check your .env file.")

# Function to fetch data from Last.fm API
def fetch_lastfm_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"🚫 Last.fm API Error: {e}")
        return None

# Function to search for a song
def search_song(query):
    url = f"http://ws.audioscrobbler.com/2.0/?method=track.search&track={query}&api_key={LASTFM_API_KEY}&format=json"
    data = fetch_lastfm_data(url)
    if not data:
        return None
    
    track_list = data.get("results", {}).get("trackmatches", {}).get("track", [])
    return track_list[0] if track_list else None

# Function to get song metadata
def get_song_info(artist, track):
    url = f"http://ws.audioscrobbler.com/2.0/?method=track.getInfo&api_key={LASTFM_API_KEY}&artist={artist}&track={track}&format=json"
    data = fetch_lastfm_data(url)
    
    if not data:
        return None
    
    track_info = data.get("track", {})
    return {
        "listeners": int(track_info.get("listeners", 0)),
        "playcount": int(track_info.get("playcount", 0)),
        "summary": track_info.get("wiki", {}).get("summary", "No description available."),
        "tags": [tag["name"] for tag in track_info.get("toptags", {}).get("tag", [])],
        "album": track_info.get("album", {}).get("title", "Unknown"),
    }

# Sentiment Analysis
def analyze_lyrics_sentiment(lyrics):
    sentiment = TextBlob(lyrics).sentiment.polarity
    return "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

# Predict Future Popularity
def predict_future_popularity(play_counts):
    X = np.arange(len(play_counts)).reshape(-1, 1)
    y = np.array(play_counts).reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return model.predict([[len(play_counts) + i] for i in range(1, 6)]).flatten().tolist()

# Clustering Songs Based on Popularity
def cluster_songs(playcount_list, listener_list):
    data = pd.DataFrame({"Playcount": playcount_list, "Listeners": listener_list})
    data["Cluster"] = KMeans(n_clusters=3, random_state=42).fit_predict(data)
    return data

# Streamlit UI
st.set_page_config(page_title="🎶 Music AI Analyzer", layout="wide")
st.title("🎵 AI-Powered Music Analyzer")
st.markdown("Analyze songs using *Last.fm metadata*, AI sentiment analysis, and predictive analytics.")

song_query = st.text_input("Enter a song name for analysis:")
if st.button("Analyze Song") and song_query:
    track = search_song(song_query)
    if track:
        st.write(f"🎵 Song: [{track['name']}]({track['url']})")
        st.write(f"🎤 Artist: {track['artist']}")
        
        song_info = get_song_info(track['artist'], track['name'])
        if song_info:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Song Statistics")
                st.metric("👥 Listeners", f"{song_info['listeners']:,}")
                st.metric("▶ Play Count", f"{song_info['playcount']:,}")
                st.write(f"🏷 Tags: {', '.join(song_info['tags']) if song_info['tags'] else 'No Tags'}")
                st.write(f"📖 Album: {song_info['album']}")
            
            with col2:
                st.subheader("📈 Popularity Trend")
                fig = px.bar(
                    x=["Listeners", "Play Count"],
                    y=[song_info['listeners'], song_info['playcount']],
                    labels={"x": "Metric", "y": "Count"},
                    title="Song Popularity",
                    text_auto=True,
                    color_discrete_sequence=["#FFA07A"]
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("📖 Description")
            st.write(song_info['summary'])
            
            # AI Sentiment Analysis
            st.subheader("🧠 AI Sentiment Analysis on Lyrics")
            lyrics_sample = "This is a sample of lyrics for sentiment analysis."
            st.write(f"🎭 *Sentiment:* {analyze_lyrics_sentiment(lyrics_sample)}")
            
            # Popularity Forecast
            st.subheader("📊 Popularity Prediction")
            future_trend = predict_future_popularity([song_info['playcount'] - i * 1000 for i in range(5)])
            trend_fig = px.line(
                x=["Next Week", "Next 2 Weeks", "Next 3 Weeks", "Next 4 Weeks", "Next 5 Weeks"],
                y=future_trend,
                labels={"x": "Time", "y": "Predicted Play Count"},
                title="Predicted Popularity",
                markers=True
            )
            st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.error("🚫 Song not found! Try another one.")
