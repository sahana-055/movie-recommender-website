import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 🎨 Custom Styled Header
st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🎬 Smart Movie Recommender</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Get intelligent suggestions from <b>Tamil</b> or <b>Hollywood</b> movies 🎭🍿</p>", unsafe_allow_html=True)
st.markdown("---")

# 🌍 Movie Source Selector
movie_type = st.selectbox("🌐 Choose Movie Type", ["Tamil", "Hollywood"])

# 📁 Load Dataset
if movie_type == "Tamil":
    movies = pd.read_csv("extended_tamil_movies.csv")
else:
    movies = pd.read_csv("movies.csv")

# 🧹 Prepare Data
movies['genres'] = movies['genres'].fillna('')
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies.index, index=movies['title'])

# 🔁 Recommendation by Movie Title
def recommend(title):
    if title not in indices:
        return ["Movie not found."]
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][['title', 'genres']]

# 🔁 Recommendation by Genre
def recommend_by_genre(selected_genre):
    filtered = movies[movies['genres'].str.contains(selected_genre, case=False, na=False)]
    return filtered[['title', 'genres']].head(10)

# ✨ Tabs for Two Recommendation Modes
tab1, tab2 = st.tabs(["🎞️ Recommend by Movie", "🎭 Recommend by Genre"])

# 🎬 Tab 1: Title-Based Recommendations
with tab1:
    st.subheader("🎯 Select a Movie You Like")
    movie_name = st.selectbox("🎞️ Choose a movie", movies['title'].values)

    if st.button("🔍 Show Similar Movies"):
        st.markdown("---")
        st.subheader("📽️ Top 10 Similar Movies")
        results = recommend(movie_name)

        for i, row in results.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.markdown(f"**🎬 {i+1}.**")
                with col2:
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"🎭 *Genres:* {row['genres']}")
            st.markdown("---")

# 🎭 Tab 2: Genre-Based Recommendations
with tab2:
    st.subheader("📚 Choose Genre You Love")
    genre_list = sorted(set(g for genre_str in movies['genres'] for g in genre_str.split('|')))
    selected_genre = st.selectbox("🎭 Select a genre", genre_list)

    if st.button("🎬 Recommend by Genre"):
        st.markdown("---")
        st.subheader(f"Top 10 🎞️ {selected_genre} Movies")
        genre_results = recommend_by_genre(selected_genre)

        for i, row in genre_results.iterrows():
            with st.container():
                col1, col2 = st.columns([1, 5])
                with col1:
                    st.markdown(f"**🎬 {i+1}.**")
                with col2:
                    st.markdown(f"**{row['title']}**")
                    st.markdown(f"🎭 *Genres:* {row['genres']}")
            st.markdown("---")
