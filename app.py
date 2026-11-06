import streamlit as st
from dotenv import load_dotenv

from rag_core import RAGError, RAGService


load_dotenv()


st.set_page_config(page_title="Document AI", page_icon="📚")
st.title("📚 Document AI")
st.caption("Ask grounded questions about your uploaded documents.")

service = RAGService.from_environment()
with st.sidebar:
    st.header("Knowledge base")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Index documents", disabled=not uploaded_files):
        try:
            with st.spinner("Indexing documents..."):
                result = service.index_uploads(uploaded_files)
            st.success(f"Indexed {result} chunk(s).")
        except RAGError as error:
            st.error(str(error))
    st.write(f"Indexed {service.count()} chunk(s)")
    st.caption(f"Answer model: {service.settings.answer_model}")

question = st.text_input("Ask a question")
if question:
    try:
        with st.spinner("Searching and generating an answer..."):
            answer, sources = service.ask(question)
        st.subheader("Answer")
        st.write(answer)
        with st.expander("Sources"):
            for source in sources:
                st.markdown(
                    f"**{source['source']}**, page {source.get('page', 'N/A')} "
                    f"(score: {source['score']:.3f})"
                )
                st.write(source["text"])
    except RAGError as error:
        st.error(str(error))
