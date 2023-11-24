from matplotlib import pyplot as plt
import streamlit as st


def train_ann_page(ann_main):
    st.subheader("Training ANN with PSO")
    plot_container = st.empty()

    for i, l, a, p in ann_main.train_ann():
        plot = plot_particles(p, i)
        plot_container.pyplot(plot)


def plot_particles(particles, iteration):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(particles)), particles, alpha=0.5)
    plt.title(f"Particle positions at iteration {iteration}")
    plt.xlabel("Particle Index")
    plt.ylabel("Particle Position")
    plt.grid(True)
    return plt
