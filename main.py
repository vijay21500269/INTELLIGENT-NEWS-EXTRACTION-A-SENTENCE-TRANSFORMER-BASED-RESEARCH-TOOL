import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import re
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import time

# Ensure NLTK punkt tokenizer is downloaded
nltk.download('punkt')

# Set up the page config for a wide display
st.set_page_config()

# Custom styles for a colorful interface
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50 !important;
        color: white !important;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 18px;
    }
    .stTextInput>div>input {
        background-color: #FFD202 !important;
        color: #236 !important;
        padding: 10px;
        border-radius: 5px;
        font-size: 16px;
    }
    .stSidebar {
        background-color: #FF6347;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize the model
model = SentenceTransformer('all-mpnet-base-v2')

# Title and Sidebar
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# Number of URLs input
url_count = st.sidebar.number_input("How many URLs do you want to add?", min_value=1, value=1, step=1)
urls = []

# Generate URL input fields based on user selection
for i in range(url_count):
    url = st.sidebar.text_input(f"URL {i + 1}", key=f"url_{i}")
    urls.append(url)

# File path for saving processed data
file_path = "faiss_store.pkl"

# Function to extract article content while preserving original capitalization
def extract_main_article_text(soup):
    main_content = (
            soup.find('article') or
            soup.find('div', {'class': ['article-content', 'main-content']}) or
            soup.find('div', text=re.compile('content', re.IGNORECASE))
    )

    # Attempt extraction from article or main content
    if main_content:
        text = ' '.join([p.get_text(" ", strip=True) for p in main_content.find_all(['p', 'h1', 'h2'])])
    else:
        # Fallback: Extract all paragraphs if specific tags are not found
        all_paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all('p')]
        filtered_paragraphs = [p for p in all_paragraphs if len(p) > 50]
        text = ' '.join(filtered_paragraphs)

    # Clean text by reducing extra spaces without converting text to lowercase
    text = re.sub(r'\s+', ' ', text)
    sentences = sent_tokenize(text)

    # If no valid content is found, return an empty list
    if not sentences or "errors.edgesuite.net" in text:
        return []

    return sentences



# Create a single "Process URLs" button
if st.sidebar.button("Process URLs"):
    if not any(urls):  # Ensure at least one URL is entered
        st.write("Please provide at least one URL to process.")
    else:
        with st.spinner("Processing URLs... Please wait!"):
            time.sleep(2)  # Simulate processing time
            valid_urls = []
            documents = []
            sentences = []  # To store individual sentences
            sentence_document_map = []

            for url in urls:
                if url:  # Only process non-empty URLs
                    try:
                        response = requests.get(url, timeout=5)

                        # Check if the content type is HTML
                        if "text/html" in response.headers.get("Content-Type", ""):
                            soup = BeautifulSoup(response.content, 'html.parser')

                            # Check if site is blocking access
                            if "security service" in soup.text.lower():
                                st.write(f"Cannot access content from {url} due to security restrictions.")
                                continue

                            text = extract_main_article_text(soup)

                            # If no content is extracted, notify the user
                            if not text:
                                st.write(
                                    f"No valid article content found in {url}. Ensure the URL has an accessible article.")
                            else:
                                documents.append(text)
                                sentences.extend(text)  # Add extracted sentences directly
                                sentence_document_map.extend([len(documents) - 1] * len(text))
                                valid_urls.append(url)
                        else:
                            st.write(f"{url} is not an HTML page. Please provide a valid article URL.")

                    except Exception as e:
                        st.write(f"Cannot process {url}. Please provide a valid and accessible URL. Error: {e}")

            if valid_urls and sentences:
                # Create embeddings for the sentences
                embeddings = model.encode(sentences, convert_to_tensor=True)

                # Save sentences, embeddings, documents, and URLs to a pickle file
                with open(file_path, "wb") as f:
                    pickle.dump((sentences, embeddings, documents, valid_urls, sentence_document_map), f)
            else:
                st.write("No valid URLs were processed. Ensure that URLs point to accessible HTML article content.")


# Sidebar button to clear existing pickle file
if st.sidebar.button("Clear Stored Data"):
    if os.path.exists(file_path):
        os.remove(file_path)
        st.sidebar.write("Stored data has been cleared.")

# Input for user query
query = st.text_input("Question : ")

# Display main content and add an expander for the graph
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            sentences, embeddings, documents, urls, sentence_document_map = pickle.load(f)
            query_embedding = model.encode(query, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0].cpu().numpy()

            # Define a threshold for cosine similarity
            similarity_threshold = 0.4  # Adjusted value for better results

            # Find the top results
            top_indices = np.argsort(cosine_scores)[-5:][::-1]  # Get indices of the top 5 scores
            top_scores = cosine_scores[top_indices]

            # Determine the best answer
            best_answer_index = None
            for rank, idx in enumerate(top_indices):
                if top_scores[rank] > similarity_threshold:  # Check if the score exceeds the threshold
                    best_answer_index = idx  # idx is the index in the original sentences array
                    break

            st.header("Answer")
            if best_answer_index is not None:
                answer_text = sentences[best_answer_index]
                answer_text = answer_text.replace('$', '&#36;')  # Encode $ for HTML
                answer_text = re.sub(r'\s+', ' ', answer_text)  # Normalize whitespace

                # Display the answer using Markdown with consistent font styling
                st.markdown(f"<p style='font-family:Roboto; font-size:16px; line-height:1.6;'>{answer_text}</p>",
                            unsafe_allow_html=True)

                # Retrieve the corresponding source URL and display it
                relevant_document_index = sentence_document_map[best_answer_index]
                if 0 <= relevant_document_index < len(urls):
                    source_url = urls[relevant_document_index]
                    st.write("Source:", source_url)
                else:
                    st.write("Source URL not found.")
            else:
                st.write(
                    "No relevant match found. URL content might not answer the question. Please rephrase or try another URL.")

            # Expander for displaying the similarity score graphs
            with st.expander("View Cosine Similarity Scores", expanded=True):
                tab1, tab2 = st.tabs(["All Scores", "Top 5 Scores"])

                with tab1:
                    fig_all, ax_all = plt.subplots(figsize=(15, 10))
                    ax_all.barh(range(len(cosine_scores)), cosine_scores, color="skyblue")
                    yticks = np.linspace(0, len(sentences) - 1, min(10, len(sentences))).astype(int)
                    ax_all.set_yticks(yticks)
                    ax_all.set_yticklabels([sentences[i] for i in yticks], fontsize=10, rotation=30)
                    ax_all.invert_yaxis()
                    ax_all.set_xlabel("Cosine Similarity Score", fontsize=14)
                    ax_all.set_title("Cosine Similarity between Query and Sentences (All)", fontsize=14)
                    ax_all.grid(axis='x')
                    st.pyplot(fig_all)

                with tab2:
                    top_sentences = [sentences[i] for i in top_indices]
                    fig_top5, ax_top5 = plt.subplots(figsize=(15, 8))
                    ax_top5.barh(range(len(top_scores)), top_scores, color="coral")
                    ax_top5.set_yticks(range(len(top_scores)))
                    ax_top5.set_yticklabels(top_sentences, fontsize=14)
                    ax_top5.invert_yaxis()
                    ax_top5.set_xlabel("Cosine Similarity Score", fontsize=14)
                    ax_top5.set_title("Top 5 Sentence Matches", fontsize=14)
                    ax_top5.grid(axis='x')
                    st.pyplot(fig_top5)
    else:
        st.write("Please process URLs first.")
