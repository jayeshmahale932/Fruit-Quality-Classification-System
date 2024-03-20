import cv2
import joblib
import numpy as np
import streamlit as st
from skimage.feature import local_binary_pattern


def load_model(model_path):
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None


def load_scaler(scaler_path):
    try:
        scaler = joblib.load(scaler_path)
        return scaler
    except Exception as e:
        st.error(f"Error loading the scaler: {e}")
        return None


def extract_color_features(image):
    hist_b = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([image], [2], None, [256], [0, 256])
    color_features = np.concatenate((hist_b, hist_g, hist_r))
    return color_features


def extract_texture_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp_image = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, 59), range=(0, 58))
    return lbp_hist


def extract_features(image):
    extracted_features = []

    image = cv2.resize(image, (128, 128))
    color_feats = extract_color_features(image)

    texture_feats = extract_texture_features(image)

    combined = np.concatenate((color_feats.ravel(), texture_feats.ravel()))
    extracted_features.append(combined)

    return extracted_features


def transform_feat(image, scaler):
    try:
        extracted_features = extract_features(image)
        image = scaler.transform(extracted_features)
        return image
    except Exception as e:
        st.error(f"Error transforming image: {e}")
        return None


def predict(model, image):
    if model is None:
        return None
    try:
        prediction = model.predict(image)
        return prediction
    except Exception as e:
        st.error(f"Error predicting: {e}")
        return None


def main():
    st.title("Fruit Quality Classification System")
    st.write("Upload an image to check if a fruit is good or bad.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if "model" not in st.session_state:
        st.session_state.model = load_model("model.h5")

    if "scaler" not in st.session_state:
        st.session_state.scaler = load_scaler("scaler.h5")

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        if image is not None:
            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                caption="Uploaded Image.",
                use_column_width=True,
            )

            if st.button("Predict"):
                feat = transform_feat(image, st.session_state.scaler)

                if feat is not None:
                    prediction = predict(st.session_state.model, feat)

                    if prediction is not None:
                        if prediction[0] == "good":
                            st.success("Your fruit has a good quality.")
                        elif prediction[0] == "bad":
                            st.warning("Your fruit has a bad quality.")
                        else:
                            st.info("Your fruit has a mixed quality.")
                    else:
                        st.error("Error in making the prediction.")
                else:
                    st.error("Error processing the image.")
        else:
            st.error("Error: Unable to read the image. Please upload a valid image.")


if __name__ == "__main__":
    main()
