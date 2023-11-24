import streamlit as st
from graphviz import Digraph


def show_ann_page(ann_main):
    st.subheader("ANN Architecture")
    layers = ann_main.get_layers()

    st.graphviz_chart(create_ann_graph(layers))

    col1, col2 = st.columns(2)
    with col1:
        back = st.button("Back: Create ANN")
    with col2:
        next = st.button("Next: Configure PSO")
    if back:
        st.session_state["page"] = "create_ann"
        st.rerun()
    if next:
        st.session_state["page"] = "configure_pso"
        st.rerun()


def create_ann_graph(layers):
    dot = Digraph(format="png")

    # Set the background color to transparent
    dot.attr(bgcolor="transparent")

    # Global graph settings
    dot.attr(
        "node",
        shape="circle",
        style="filled",
        fillcolor="lightblue",
        fontname="Helvetica",
    )
    dot.attr("edge", arrowhead="vee", arrowsize="0.5", color="white")
    dot.attr(rankdir="LR")

    # Define nodes and edges based on layers
    for i, layer in enumerate(layers, start=1):
        with dot.subgraph(name=f"cluster_{i}") as c:
            c.attr(color="transparent", fontname="Helvetica", fontcolor="white" )
            c.attr(label=f'Layer {i}\nActivation: {layer["activation"]}')

            # Add nodes for the current layer
            for n in range(layer["nodes"]):
                node_name = (
                    f"L{i}_N{n+1}"  # Node names (e.g., L1_N1 for Layer 1, Node 1)
                )
                c.node(node_name, label=f"Node {n+1}")

            # Connect nodes to previous layer if not the first layer
            if i > 1:
                for prev_n in range(layers[i - 2]["nodes"]):
                    for curr_n in range(layer["nodes"]):
                        # Connection from all nodes in previous layer to current
                        dot.edge(f"L{i-1}_N{prev_n+1}", f"L{i}_N{curr_n+1}")

    return dot
