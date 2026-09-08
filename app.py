import os
import re
import hashlib

import streamlit as st
import fitz  # PyMuPDF
import faiss

from groq import Groq
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "openai/gpt-oss-20b"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 5


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="👩‍💼",
    layout="wide",
)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


embedding_model = load_embedding_model()
tokenizer = load_tokenizer()


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    api_key = None

    try:
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


groq_client = get_groq_client()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(pdf_bytes):

    document = fitz.open(
        stream=pdf_bytes,
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text("text")

        if text.strip():

            pages.append({
                "page": page_number + 1,
                "text": text
            })

    document.close()

    return pages


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    return text.strip()


# ============================================================
# TOKEN-AWARE CHUNKING
# ============================================================

def create_chunks(pages):

    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for page_data in pages:

        page_number = page_data["page"]

        text = clean_text(
            page_data["text"]
        )

        if not text:
            continue

        token_ids = tokenizer.encode(
            text,
            add_special_tokens=False
        )

        start = 0

        while start < len(token_ids):

            end = start + CHUNK_SIZE

            chunk_token_ids = token_ids[start:end]

            chunk_text = tokenizer.decode(
                chunk_token_ids,
                skip_special_tokens=True
            ).strip()

            if chunk_text:

                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            if end >= len(token_ids):
                break

            start += step

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.astype("float32")


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(
    question,
    index,
    chunks,
    top_k=TOP_K
):

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        result = chunks[index_id].copy()

        result["score"] = float(score)

        results.append(result)

    return results


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_chunks,
    client
):

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks
    ):

        context_parts.append(
            f"""
SOURCE {i + 1}
Page: {chunk['page']}

{chunk['text']}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
You are an HR Policy Assistant.

Answer the user's question ONLY using
the provided HR policy context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available in the
   provided context, say:

"I could not find this information in
the uploaded HR policy."

4. Keep the answer clear and concise.
5. When possible, mention the relevant
   policy page.
6. If the question is unrelated to the
   uploaded HR policy, politely explain
   that you can only answer questions
   based on the uploaded HR policy.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

USER QUESTION:

{question}

Provide a grounded answer based only
on the HR policy context.
"""

    response = client.chat.completions.create(

        model=LLM_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.1,

        max_completion_tokens=800
    )

    return response.choices[0].message.content


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("👩‍💼 HR Policy Assistant")

st.write(
    "Upload an HR policy PDF and ask "
    "questions about its contents."
)

st.caption(
    "RAG Pipeline: PDF → Chunks → Embeddings "
    "→ FAISS → Retrieval → Groq → Answer"
)


# ============================================================
# API KEY STATUS
# ============================================================

if groq_client is None:

    st.warning(
        "GROQ_API_KEY is not configured. "
        "Add it to Streamlit Cloud Secrets "
        "before asking questions."
    )


# ============================================================
# SESSION STATE
# ============================================================

if "index" not in st.session_state:
    st.session_state.index = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "file_hash" not in st.session_state:
    st.session_state.file_hash = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload HR Policy PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    current_hash = hashlib.md5(
        pdf_bytes
    ).hexdigest()

    if current_hash != st.session_state.file_hash:

        with st.spinner(
            "Processing HR policy PDF..."
        ):

            pages = extract_pdf_text(
                pdf_bytes
            )

            if not pages:

                st.error(
                    "No extractable text was "
                    "found in this PDF. It may "
                    "be a scanned/image-only PDF."
                )

                st.stop()

            chunks = create_chunks(
                pages
            )

            if not chunks:

                st.error(
                    "Could not create text "
                    "chunks from the PDF."
                )

                st.stop()

            embeddings = create_embeddings(
                chunks
            )

            index = build_faiss_index(
                embeddings
            )

            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.file_hash = current_hash
            st.session_state.file_name = (
                uploaded_file.name
            )

        st.success(
            f"Policy processed successfully: "
            f"{len(pages)} pages and "
            f"{len(chunks)} chunks created."
        )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if (
    st.session_state.index is not None
    and st.session_state.chunks is not None
):

    st.subheader(
        "Document Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        page_count = len({
            chunk["page"]
            for chunk in st.session_state.chunks
        })

        st.metric(
            "Pages",
            page_count
        )

    with col2:

        st.metric(
            "Chunks",
            len(st.session_state.chunks)
        )

    with col3:

        st.metric(
            "Top-K Sources",
            TOP_K
        )

    st.info(
        f"Current document: "
        f"{st.session_state.file_name}"
    )


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

with st.sidebar:

    st.header(
        "Example Questions"
    )

    st.write(
        "Try asking:"
    )

    st.write(
        "• How many annual leave days "
        "are employees entitled to?"
    )

    st.write(
        "• Can employees work remotely?"
    )

    st.write(
        "• What are the standard "
        "working hours?"
    )

    st.write(
        "• How many sick leave days "
        "are available?"
    )

    st.write(
        "• What is the resignation "
        "notice period?"
    )

    st.write(
        "• What is the employee salary?"
    )


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the "
    "uploaded HR policy..."
)


# ============================================================
# ANSWER
# ============================================================

if question:

    if groq_client is None:

        st.error(
            "GROQ_API_KEY is missing. "
            "Configure it in Streamlit Secrets."
        )

        st.stop()

    if st.session_state.index is None:

        st.error(
            "Please upload an HR policy PDF first."
        )

        st.stop()

    with st.spinner(
        "Searching the HR policy..."
    ):

        retrieved_chunks = retrieve_chunks(
            question,
            st.session_state.index,
            st.session_state.chunks,
            TOP_K
        )

    with st.spinner(
        "Generating grounded answer..."
    ):

        answer = generate_answer(
            question,
            retrieved_chunks,
            groq_client
        )

    st.subheader("Answer")

    st.write(answer)

    st.subheader(
        "Retrieved Policy Sources"
    )

    for i, chunk in enumerate(
        retrieved_chunks
    ):

        with st.expander(
            f"Source {i + 1} — "
            f"Page {chunk['page']} — "
            f"Similarity {chunk['score']:.3f}"
        ):

            st.write(
                chunk["text"]
            )
