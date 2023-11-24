import streamlit as st


def configure_pso_page(ann_main):
    st.subheader("Configure PSO")

    swarm_size = st.number_input("Swarm Size", min_value=1, value=10)
    informants_type = st.radio(
        "Informants Type",
        options=["Random", "Neighbours", "Global Best and Neighbours"],
    )
    optimize_activation = st.toggle("Optimize Activation", value=False)
    opt = st.radio("Optimization", options=["Maximize", "Minimize"])
    informants_size = st.number_input("Informants Size", min_value=1, value=3)
    iterations = st.number_input("Iterations", min_value=1, value=100)

    alpha = st.number_input(
        "Alpha",
        help="Inertia Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
    )
    beta = st.number_input(
        "Beta",
        help="Cognitive Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
    )
    gamma = st.number_input(
        "Gamma", help="Social Weight", min_value=0.0, max_value=1.0, value=0.0, step=0.1
    )
    delta = st.number_input(
        "Delta", help="Global Weight", min_value=0.0, max_value=1.0, value=0.0, step=0.1
    )

    epsilon = st.number_input(
        "Epsilon",
        help="position update step size",
        min_value=0.4,
        max_value=0.9,
        value=0.4,
        step=0.1,
    )

    randomize_weights = st.checkbox("Randomize Weights and Epsilon", value=False)

    col1, col2 = st.columns(2)
    with col1:
        back = st.button("Back: Show ANN")
    with col2:
        next = st.button("Next: Train ANN")
    if back:
        st.session_state["page"] = "show_ann"
        st.rerun()
    if next:
        if beta + gamma + delta != 4 and not randomize_weights:
            st.error("Sum of Beta, Gamma and Delta must be 4")
            return

        ann_main.set_pso(
            randomize_weights,
            swarm_size,
            informants_type,
            informants_size,
            iterations,
            optimize_activation,
            opt,
            alpha,
            beta,
            gamma,
            delta,
            epsilon,
        )
        st.session_state["page"] = "train_ann"
        st.rerun()
