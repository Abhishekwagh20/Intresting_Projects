# vector_store.py
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Dict, Tuple

class VectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the vector store with a pre-trained Sentence Transformer model.
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)  # L2 distance for similarity
        self.file_mappings = {}  # Maps index to file path and content
        self.current_index = 0

    def add_documents(self, documents: Dict[str, str]):
        """
        Add documents to the vector store.
        :param documents: Dictionary of {file_path: file_content}
        """
        for file_path, content in documents.items():
            # Split content into chunks (e.g., by lines or functions)
            chunks = self._split_into_chunks(content)
            for chunk in chunks:
                # Generate embedding for each chunk
                embedding = self.model.encode(chunk, convert_to_tensor=False)
                embedding = np.array(embedding).astype("float32")
                
                # Add to FAISS index
                self.index.add(np.array([embedding]))
                
                # Store metadata
                self.file_mappings[self.current_index] = {
                    "file_path": file_path,
                    "content": chunk
                }
                self.current_index += 1

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for the most relevant code chunks.
        :param query: User query
        :param top_k: Number of results to return
        :return: List of relevant code chunks with metadata
        """
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=False)
        query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)

        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)

        # Retrieve results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx in self.file_mappings:
                result = self.file_mappings[idx]
                result["score"] = float(1 / (1 + distance))  # Convert distance to similarity score
                results.append(result)

        return results

    def _split_into_chunks(self, content: str, max_chunk_size: int = 512) -> List[str]:
        """
        Split content into smaller chunks for better retrieval.
        :param content: File content
        :param max_chunk_size: Maximum number of characters per chunk
        :return: List of chunks
        """
        lines = content.split("\n")
        chunks = []
        current_chunk = ""

        for line in lines:
            if len(current_chunk) + len(line) <= max_chunk_size:
                current_chunk += line + "\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = line + "\n"

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks