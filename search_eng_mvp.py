import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os


def main():
    st.title("Simple FAISS Search App")

    # 1. File uploaders: CSV and FAISS index
    csv_file = st.file_uploader("Upload your CSV file", type=["csv"])
    index_file = st.file_uploader("Upload your FAISS .index file", type=["index"])

    if csv_file is not None and index_file is not None:
        # 2. Load DataFrame
        df = pd.read_csv(csv_file)
        st.write("Preview of uploaded CSV:")
        st.write(df.head())

        # 3. Save the index file to a temporary location and load the FAISS index
        with open("temp.index", "wb") as f:
            f.write(index_file.getbuffer())
        faiss_index = faiss.read_index("temp.index")

        # 4. Load the SentenceTransformer model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # 5. Input for user query
        query = st.text_input("Enter your query:")
        if st.button("Search") and query.strip() != "":
            # 6. Encode the query
            query_vector = model.encode(query).reshape(1, -1)

            # 7. Perform the search
            k = 5  # number of results to retrieve
            distances, indices = faiss_index.search(query_vector, k)

            # 8. Display results
            st.write("**Top Results:**")
            for rank, idx in enumerate(indices[0]):
                st.markdown(f"**Result {rank+1}:**")
                st.write(f"- **Ref**: {df.iloc[idx]['ref']}")
                # Uncomment or add additional fields as desired:
                # st.write(f"- Brand: {df.iloc[idx]['Brand']}")
                # st.write(f"- Description: {df.iloc[idx]['itemdescriptionEN']}")
                st.write("---")


if __name__ == "__main__":
    main()
