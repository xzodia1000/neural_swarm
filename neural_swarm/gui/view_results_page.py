import streamlit as st
import matplotlib.pyplot as plt

from neural_swarm.ann_main import ANNMain


def view_results_page(ann_main):
    st.subheader("Results")

    train_loss = ann_main.ann_loss
    train_accuracy = ann_main.ann_acc
    test_accuracy, test_loss = ann_main.evaluate_results()
    epochs = range(1, ann_main.iterations + 1)

    st.markdown(
        f"<div style='color: green; font-size: 20px;'>Training Loss: {train_loss[-1]}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='color: green; font-size: 20px;'>Training Accuracy: {train_accuracy[-1]}%</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='color: red; font-size: 20px;'>Testing Loss: {test_loss}</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"<div style='color: red; font-size: 20px;'>Testing Accuracy: {test_accuracy}%</div>",
        unsafe_allow_html=True,
    )

    # Plot for Loss
    plt.figure()
    plt.plot(epochs, train_loss, label="Loss", color="red")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    st.pyplot(plt)  # Display the loss plot

    # Plot for Accuracy
    plt.figure()
    plt.plot(epochs, train_accuracy, label="Accuracy", color="blue")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training Accuracy")
    plt.legend()
    st.pyplot(plt)  # Display the accuracy plot

    if st.button("Next: Restart"):
        st.session_state["page"] = "home"
        st.session_state["ann_main"] = ANNMain()
        st.rerun()
