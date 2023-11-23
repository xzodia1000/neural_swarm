import random
import streamlit as st


def create_ann_page(ann_main):
    st.subheader("Create an ANN")

    input_nodes = ann_main.first_layer
    output_nodes = ann_main.last_layer
    output_activation = ann_main.last_layer_activation.__str__()

    activation_functions_map = {
        "Sigmoid": 0,
        "Tanh": 1,
        "Relu": 2,
        "Softmax": 3,
        "Identity": 4,
    }

    if "layers" not in st.session_state:
        st.session_state.layers = []
        add_layer(input_nodes, index=0)
        add_layer(
            output_nodes,
            activation_functions_map[output_activation],
            index=len(st.session_state.layers) + 1,
        )

    if "new_layer" not in st.session_state:
        st.session_state.new_layer = True

    if st.session_state.new_layer:
        st.session_state.new_layer = False
        show_layers()

    col1, col2 = st.columns(2)
    with col1:
        back = st.button("Back", key="back4")
    with col2:
        next = st.button("Next", key="next4")
    if back:
        st.session_state["page"] = "view_dataset"
        st.rerun()
    if next:
        st.session_state["page"] = "show_ann"
        st.rerun()


def add_layer(nodes=1, activation=0, index=-1):
    st.session_state.new_layer = True
    if st.button("Add Layer", key=f"addL{random.randint(0, 1000)}"):
        st.session_state.layers.insert(
            index, {"nodes": nodes, "activation": activation}
        )


def show_layers():
    activation_options = ["Sigmoid", "Tanh", "Relu", "Softmax", "Identity"]
    with st.container():
        for i, layer in enumerate(st.session_state.layers):
            col1, col2 = st.columns(2)

            if i == 0:
                with col1:
                    st.number_input(
                        "Number of nodes",
                        min_value=1,
                        value=layer["nodes"],
                        key=f"{i}nodes",
                        disabled=True,
                        on_change=lambda x: update_layer_nodes(i, x),
                    )
                with col2:
                    st.selectbox(
                        "Activation Function",
                        options=activation_options,
                        index=0,
                        key=f"{i}act",
                        disabled=True,
                        on_change=lambda x: update_layer_act(i, x),
                    )

            elif i != len(st.session_state.layers) - 1:
                with col1:
                    st.number_input(
                        "Number of nodes",
                        min_value=1,
                        value=layer["nodes"],
                        key=f"{i}nodes",
                        on_change=lambda x: update_layer_nodes(i, x),
                    )
                with col2:
                    st.selectbox(
                        "Activation Function",
                        options=activation_options,
                        index=activation_options.index(layer["activation"]),
                        key=f"{i}act",
                        on_change=lambda x: update_layer_act(i, x),
                    )

        add_layer()

        col1, col2 = st.columns(2)
        i = len(st.session_state.layers) - 1
        with col1:
            st.number_input(
                "Number of nodes",
                min_value=1,
                value=st.session_state.layers[i]["nodes"],
                key=f"{i}nodes",
                disabled=True,
                on_change=lambda x: update_layer_nodes(i, x),
            )
        with col2:
            st.selectbox(
                "Activation Function",
                options=activation_options,
                index=activation_options.index(
                    st.session_state.layers[i]["activation"]
                ),
                key=f"{i}act",
                disabled=True,
                on_change=lambda x: update_layer_act(i, x),
            )


def update_layer_nodes(index, value):
    st.session_state.layers[index]["nodes"] = value


def update_layer_act(index, value):
    st.session_state.layers[index]["activation"] = value
