import streamlit as st
from fastai.vision.all import *
import urllib.request

st.title("Cat vs. Dog Classifier")
st.text("Built by Lawrence Lam")


def label_func(f): return f[0].isupper()


model = load_learner('my_model.pkl')


def predict(image):
    img = PILImage.create(image)
    pred_class, pred_idx, outputs = model.predict(img)
    print(outputs)

    likelihood_is_cat = outputs[1].item()

    if likelihood_is_cat > 0.98:
        return "cat"
    elif likelihood_is_cat < 0.02:
        return "dog"
    else:
        return "Not sure... true another picture"


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        prediction = predict
        st.write(prediction)