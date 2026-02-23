"""
Day 05 — Document Q&A: Data Preparation Pipeline
Downloads sample documents and prepares them for indexing.
"""

import json
import requests
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ─── Sample Documents ────────────────────────────────────────────────
SAMPLE_DOCS = [
    {
        "title": "Python Programming Language",
        "content": (
            "Python is a high-level, general-purpose programming language. "
            "Its design philosophy emphasizes code readability with the use of significant indentation. "
            "Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, "
            "including structured, object-oriented, and functional programming. "
            "Python was conceived in the late 1980s by Guido van Rossum at Centrum Wiskunde & Informatica "
            "in the Netherlands as a successor to the ABC programming language. "
            "Python consistently ranks as one of the most popular programming languages. "
            "It is used in web development, data science, artificial intelligence, scientific computing, "
            "and many other domains. Popular frameworks include Django, Flask, FastAPI for web development, "
            "and TensorFlow, PyTorch, scikit-learn for machine learning. "
            "Python 3.12, released in October 2023, introduced several performance improvements "
            "and new syntax features including improved error messages and a new type parameter syntax."
        ),
    },
    {
        "title": "Machine Learning Fundamentals",
        "content": (
            "Machine learning is a subset of artificial intelligence that focuses on building systems "
            "that learn from data. There are three main types of machine learning: supervised learning, "
            "unsupervised learning, and reinforcement learning. "
            "Supervised learning uses labeled training data to learn a mapping from inputs to outputs. "
            "Common algorithms include linear regression, decision trees, random forests, and neural networks. "
            "Unsupervised learning finds hidden patterns in unlabeled data. Techniques include clustering "
            "(K-means, DBSCAN), dimensionality reduction (PCA, t-SNE), and association rule learning. "
            "Reinforcement learning trains agents to make sequential decisions by maximizing cumulative reward. "
            "Deep learning, a subset of ML using neural networks with multiple layers, has achieved "
            "breakthrough results in image recognition, natural language processing, and game playing. "
            "Key concepts include overfitting, underfitting, bias-variance tradeoff, cross-validation, "
            "feature engineering, and hyperparameter tuning. "
            "Popular tools include scikit-learn, TensorFlow, PyTorch, XGBoost, and Hugging Face Transformers."
        ),
    },
    {
        "title": "Natural Language Processing",
        "content": (
            "Natural Language Processing (NLP) is a field of AI that deals with the interaction between "
            "computers and human language. Key NLP tasks include text classification, named entity recognition, "
            "sentiment analysis, machine translation, question answering, and text summarization. "
            "Traditional NLP relied on rule-based and statistical methods. Modern NLP is dominated by "
            "transformer-based models like BERT, GPT, T5, and their variants. "
            "The transformer architecture, introduced in the 2017 paper 'Attention Is All You Need', "
            "uses self-attention mechanisms to process sequential data in parallel. "
            "BERT (Bidirectional Encoder Representations from Transformers) excels at understanding tasks "
            "like classification and question answering. GPT (Generative Pre-trained Transformer) excels "
            "at text generation. T5 treats every NLP problem as a text-to-text problem. "
            "Word embeddings like Word2Vec, GloVe, and FastText represent words as dense vectors. "
            "Tokenization, stemming, lemmatization, and stop word removal are common preprocessing steps. "
            "Libraries like spaCy, NLTK, Hugging Face Transformers, and Gensim are widely used."
        ),
    },
    {
        "title": "Vector Databases and Similarity Search",
        "content": (
            "Vector databases are specialized database systems designed to store, index, and query "
            "high-dimensional vector embeddings. They enable fast similarity search, which is crucial "
            "for applications like recommendation systems, image search, and retrieval-augmented generation. "
            "Popular vector databases include FAISS (Facebook AI Similarity Search), Pinecone, Weaviate, "
            "Milvus, Qdrant, and ChromaDB. "
            "FAISS is an open-source library developed by Meta AI Research. It supports multiple index types "
            "including Flat (brute-force), IVF (inverted file), HNSW (hierarchical navigable small world), "
            "and PQ (product quantization). "
            "Similarity metrics include cosine similarity, Euclidean distance (L2), and inner product. "
            "Approximate Nearest Neighbor (ANN) search trades a small amount of accuracy for significantly "
            "faster query times. This is essential when dealing with millions or billions of vectors. "
            "The typical workflow involves: generating embeddings from raw data using a model, "
            "indexing the embeddings in a vector database, and querying with a new embedding to find "
            "the most similar stored vectors."
        ),
    },
    {
        "title": "FastAPI Web Framework",
        "content": (
            "FastAPI is a modern, fast web framework for building APIs with Python based on standard "
            "Python type hints. It is one of the fastest Python frameworks available, on par with "
            "NodeJS and Go. FastAPI is built on top of Starlette for web parts and Pydantic for data "
            "validation. Key features include automatic API documentation (Swagger UI and ReDoc), "
            "built-in data validation, dependency injection, and async support. "
            "FastAPI uses Python type hints to automatically validate request data, serialize responses, "
            "and generate OpenAPI documentation. It supports WebSockets, background tasks, middleware, "
            "CORS, and static files. "
            "A typical FastAPI application uses path operations (GET, POST, PUT, DELETE), "
            "request body models defined with Pydantic, query parameters, path parameters, "
            "and dependency injection for shared logic. "
            "FastAPI supports async/await for non-blocking I/O operations, making it ideal for "
            "high-performance APIs that need to handle many concurrent requests."
        ),
    },
    {
        "title": "Docker Containerization",
        "content": (
            "Docker is a platform for developing, shipping, and running applications in containers. "
            "Containers are lightweight, portable, and self-sufficient units that package an application "
            "with all its dependencies. Unlike virtual machines, containers share the host OS kernel, "
            "making them more efficient. "
            "A Dockerfile defines the steps to build a container image. Common instructions include "
            "FROM (base image), RUN (execute commands), COPY (copy files), WORKDIR (set directory), "
            "EXPOSE (declare ports), and CMD (default command). "
            "Docker Compose is a tool for defining multi-container applications using a YAML file. "
            "It manages service dependencies, networks, volumes, and environment variables. "
            "Best practices include using multi-stage builds to reduce image size, using .dockerignore "
            "to exclude unnecessary files, running as non-root user, and using health checks. "
            "Docker Hub is the default public registry for container images. Private registries include "
            "AWS ECR, Google Container Registry, and Azure Container Registry."
        ),
    },
    {
        "title": "Retrieval-Augmented Generation (RAG)",
        "content": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval "
            "with text generation to produce more accurate and grounded responses. Instead of relying "
            "solely on a language model's parametric knowledge, RAG retrieves relevant documents from "
            "an external knowledge base and uses them as context for generation. "
            "The RAG pipeline typically consists of three stages: indexing, retrieval, and generation. "
            "During indexing, documents are chunked, embedded, and stored in a vector database. "
            "During retrieval, the user query is embedded and used to find the most similar document chunks. "
            "During generation, the retrieved chunks are passed as context to a language model. "
            "Benefits of RAG include reduced hallucination, up-to-date knowledge, source attribution, "
            "and domain-specific expertise without fine-tuning. "
            "Common chunking strategies include fixed-size chunking, sentence-based chunking, "
            "paragraph-based chunking, and recursive character splitting. "
            "Evaluation metrics for RAG include retrieval precision, recall, MRR (Mean Reciprocal Rank), "
            "and answer accuracy measures like F1 score and exact match."
        ),
    },
    {
        "title": "Transformers Architecture",
        "content": (
            "The Transformer architecture revolutionized NLP and has since been applied to computer vision, "
            "audio processing, and other domains. It was introduced in the landmark paper 'Attention Is "
            "All You Need' by Vaswani et al. in 2017. "
            "The core innovation is the self-attention mechanism, which allows the model to weigh the "
            "importance of different positions in the input sequence when computing representations. "
            "Multi-head attention runs multiple attention operations in parallel, allowing the model "
            "to attend to information from different representation subspaces. "
            "The architecture consists of an encoder and decoder, each with multiple layers. "
            "Each layer contains multi-head attention, feed-forward networks, layer normalization, "
            "and residual connections. Positional encoding is added to give the model information "
            "about token positions. "
            "Encoder-only models (BERT) are good for understanding tasks. Decoder-only models (GPT) "
            "are good for generation. Encoder-decoder models (T5, BART) handle sequence-to-sequence tasks. "
            "Scaling laws show that larger models trained on more data consistently perform better."
        ),
    },
]


def prepare_sample_data() -> Path:
    """Create sample document files for the QA system."""
    logger.info("Preparing sample documents...")

    output_file = config.PROCESSED_DIR / "documents.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for i, doc in enumerate(SAMPLE_DOCS):
            # Save as JSONL
            record = {
                "id": f"doc_{i:03d}",
                "title": doc["title"],
                "content": doc["content"],
                "source": "sample_data",
            }
            f.write(json.dumps(record) + "\n")

            # Also save individual text files
            txt_path = config.RAW_DIR / f"{doc['title'].lower().replace(' ', '_')}.txt"
            txt_path.write_text(
                f"# {doc['title']}\n\n{doc['content']}", encoding="utf-8"
            )

    logger.success(f"Prepared {len(SAMPLE_DOCS)} sample documents → {output_file}")
    return output_file


def load_documents(path: Path | None = None) -> list[dict]:
    """Load documents from JSONL file."""
    if path is None:
        path = config.PROCESSED_DIR / "documents.jsonl"

    if not path.exists():
        logger.warning("No documents found. Preparing sample data...")
        prepare_sample_data()

    documents = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                documents.append(json.loads(line))

    logger.info(f"Loaded {len(documents)} documents from {path}")
    return documents


if __name__ == "__main__":
    prepare_sample_data()
    docs = load_documents()
    for doc in docs:
        print(f"  📄 {doc['title']} ({len(doc['content'])} chars)")
