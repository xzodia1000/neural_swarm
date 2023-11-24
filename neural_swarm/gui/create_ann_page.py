import streamlit as st


def create_ann_page(ann_main):
    st.subheader("Create an ANN")

    input_nodes = ann_main.first_layer
    output_nodes = ann_main.last_layer
    output_activation = ann_main.last_layer_activation.__str__()
    loss = ann_main.loss.__str__()

    activation_options = ["Sigmoid", "Tanh", "Relu", "Softmax", "Identity"]

    if "layers" not in st.session_state:
        st.session_state.layers = [
            {"nodes": input_nodes, "activation": 0},  # Input layer
            {
                "nodes": output_nodes,
                "activation": activation_options.index(output_activation),
            },  # Output layer
        ]

    st.markdown("#### Problem Type and Loss")
    show_loss(loss)

    st.markdown("#### ANN Architecture")
    show_layers(activation_options=activation_options)
    update_all_layers(activation_options=activation_options)

    col1, col2 = st.columns(2)
    with col1:
        back = st.button("Back: View Dataset")
    with col2:
        next = st.button("Next: Show ANN")
    if back:
        st.session_state["page"] = "view_dataset"
        st.rerun()
    if next:
        ann_main.set_ann(st.session_state.layers)
        ann_main.update_loss(st.session_state.loss)
        st.session_state["page"] = "show_ann"
        st.rerun()


def add_layer():
    if st.button("Add Hidden Layer"):
        st.session_state.layers.insert(-1, {"nodes": 1, "activation": 0})
        st.rerun()


def show_layers(activation_options):
    with st.container():
        for i, layer in enumerate(st.session_state.layers):
            col1, col2, col3 = st.columns([3, 3, 2])

            # Input layer
            if i == 0:
                with col1:
                    st.number_input(
                        "Number of Nodes",
                        min_value=1,
                        value=layer["nodes"],
                        key=f"{i}nodes",
                        disabled=True,
                    )
                with col2:
                    st.selectbox(
                        "Activation Function",
                        options=activation_options,
                        index=layer["activation"],
                        key=f"{i}act",
                    )

            # Output layer
            elif i == len(st.session_state.layers) - 1:
                with col1:
                    st.number_input(
                        "Number of Nodes",
                        min_value=1,
                        value=layer["nodes"],
                        key=f"{i}nodes",
                        disabled=True,
                    )
                with col2:
                    st.selectbox(
                        "Activation Function",
                        options=activation_options,
                        index=layer["activation"],
                        key=f"{i}act",
                        disabled=True,
                    )

            # Hidden layers
            else:
                with col1:
                    st.number_input(
                        "Number of Nodes",
                        min_value=1,
                        value=layer["nodes"],
                        key=f"{i}nodes",
                    )
                with col2:
                    st.selectbox(
                        "Activation Function",
                        options=activation_options,
                        index=layer["activation"],
                        key=f"{i}act",
                    )
                with col3:
                    st.markdown(
                        f"<div style='height: {29}px;'></div>",
                        unsafe_allow_html=True,
                    )
                    if st.button("Remove Layer", key=f"remove{i}"):
                        st.session_state.layers.pop(i)
                        st.rerun()

    add_layer()


def show_loss(loss):
    col1, col2 = st.columns(2)
    top = 29
    text_color = "green"
    text_size = "20px"

    if loss == "Mse":
        with col1:
            st.markdown(
                f"<div style='margin-top: {top}px; color: {text_color}; font-size: {text_size};'>Regression</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.selectbox(
                "Loss Function",
                options=["Mean Squared Error"],
                index=0,
                key="loss",
                disabled=True,
            )
    elif loss == "BinaryCrossEntropy":
        with col1:
            st.markdown(
                f"<div style='margin-top: {top}px; color: {text_color}; font-size: {text_size};'>Binary Classification</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.selectbox(
                "Loss Function",
                options=["Binary Cross Entropy", "Hinge"],
                index=0,
                key="loss",
            )
    elif loss == "CategoricalCrossEntropy":
        with col1:
            st.markdown(
                f"<div style='margin-top: {top}px; color: {text_color}; font-size: {text_size};'>Multi-Class Classification</div>",
                unsafe_allow_html=True,
            )
        with col2:
            st.selectbox(
                "Loss Function",
                options=["Categorical Cross Entropy"],
                index=0,
                key="loss",
                disabled=True,
            )


def update_all_layers(activation_options):
    for i, layer in enumerate(st.session_state.layers):
        layer["nodes"] = st.session_state[f"{i}nodes"]
        layer["activation"] = activation_options.index(st.session_state[f"{i}act"])
