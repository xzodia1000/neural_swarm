import streamlit as st


def train_ann_page(ann_main):
    st.subheader("Training ANN with PSO")
    progress_bar = st.progress(0)
    total_iterations = ann_main.iterations

    if "train_ann" not in st.session_state:
        st.session_state["train_ann"] = False

    if not st.session_state["train_ann"]:
        for i in ann_main.train_ann():
            progress_bar.progress(int(100 * i / total_iterations))

    st.session_state["train_ann"] = True

    if st.button("Next: View Results"):
        st.session_state["page"] = "view_results"
        del st.session_state["train_ann"]
        st.rerun()
