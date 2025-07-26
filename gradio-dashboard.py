import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr
import torch

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()

text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  # Force CPU to avoid CUDA issues
    encode_kwargs={'normalize_embeddings': True}
)

db_books = Chroma.from_documents(
    documents,
    embedding=embeddings
)

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs

def recommend_books(query: str, category: str, tone: str):
    try:
        recommendations = retrieve_semantic_recommendations(query, category, tone)
        results = []

        for _, row in recommendations.iterrows():
            description = str(row["description"]) if not pd.isna(row["description"]) else "No description available"
            truncated_desc_split = description.split()
            truncated_description = " ".join(truncated_desc_split[:30]) + "..."

            authors = str(row["authors"]) if not pd.isna(row["authors"]) else "Unknown Author"
            authors_split = authors.split(";")
            if len(authors_split) == 2:
                authors_str = f"{authors_split[0]} and {authors_split[1]}"
            elif len(authors_split) > 2:
                authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
            else:
                authors_str = authors

            title = str(row["title"]) if not pd.isna(row["title"]) else "Unknown Title"
            caption = f"{title} by {authors_str}: {truncated_description}"
            
            # Ensure we have a valid image URL
            image_url = row["large_thumbnail"] if not pd.isna(row["large_thumbnail"]) else "cover-not-found.jpg"
            results.append((image_url, caption))
        
        return results
    except Exception as e:
        print(f"Error in recommend_books: {e}")
        return [("cover-not-found.jpg", f"Error: {str(e)}")]

categories = ["All"] + sorted(books["simple_categories"].unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Create the interface with simpler components
with gr.Blocks() as dashboard:  # Removed theme to avoid potential issues
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness"
        )
        category_dropdown = gr.Dropdown(
            choices=categories, 
            label="Select a category:", 
            value="All"
        )
        tone_dropdown = gr.Dropdown(
            choices=tones, 
            label="Select an emotional tone:", 
            value="All"
        )
        submit_button = gr.Button("Find recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(
        label="Recommended books", 
        columns=4,  # Reduced columns
        rows=2
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

# Launch with additional parameters to avoid API issues
dashboard.launch(
    share=True,
    show_api=False,  # Disable API documentation
    quiet=True       # Reduce verbose output
)