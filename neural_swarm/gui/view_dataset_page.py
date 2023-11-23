import streamlit as st


def view_dataset_page(ann_main):
    st.subheader("Feature Selection and Splitting")

    st.dataframe(ann_main.dataset)

    df_columns = ann_main.get_dataset_features()

    selected_columns = st.multiselect(
        "Select features you want to keep:", df_columns, default=df_columns
    )

    new_df = ann_main.dataset[selected_columns + [ann_main.dataset.columns[-1]]]

    train_size = st.slider(
        "Select the train set size ratio",
        min_value=10,
        max_value=90,
        value=80,
        step=5,
        format="%d%%",
    )

    col1, col2 = st.columns(2)
    with col1:
        back = st.button("Back", key="back2")
    with col2:
        next = st.button("Next", key="next2")
    if back:
        st.session_state["page"] = "home"
        st.rerun()
    if next:
        st.session_state["page"] = "create_ann"
        ann_main.set_dataset(new_df)
        ann_main.set_inital_values(test_size=(100 - train_size) / 100)
        st.rerun()
