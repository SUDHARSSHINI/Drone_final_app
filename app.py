# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
@st.cache
def load_data():
    df = pd.read_csv("Crop_recommendation.csv")
    return df

# Train the model
@st.cache(allow_output_mutation=True)
def train_model():
    df = load_data()
    X = df.drop('label', axis=1)
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return rf_model, scaler, accuracy

# Main Streamlit app
def main():
    st.title("Crop Recommendation System")

    # Sidebar for user inputs
    st.sidebar.title("Enter Soil and Weather Parameters")
    
    N = st.sidebar.slider("Nitrogen (N)", 0, 140, 90)
    P = st.sidebar.slider("Phosphorus (P)", 0, 145, 40)
    K = st.sidebar.slider("Potassium (K)", 0, 205, 40)
    temperature = st.sidebar.slider("Temperature (°C)", 10, 50, 20)
    humidity = st.sidebar.slider("Humidity (%)", 10, 100, 80)
    ph = st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5)
    rainfall = st.sidebar.slider("Rainfall (mm)", 10, 300, 200)

    # Display sample dataset and accuracy
    st.subheader("Sample Dataset")
    df = load_data()
    st.write(df.head())

    rf_model, scaler, accuracy = train_model()

    st.subheader(f"Model Accuracy: {accuracy * 100:.2f}%")

    # Predict crop based on user input
    if st.sidebar.button("Recommend Crop"):
        input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
        input_data = scaler.transform(input_data)
        predicted_crop = rf_model.predict(input_data)
        st.write(f"Recommended Crop: **{predicted_crop[0]}**")

    # Feature importance plot
    if st.sidebar.checkbox("Show Feature Importance"):
        importances = rf_model.feature_importances_
        feature_names = df.drop('label', axis=1).columns
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=feature_names, ax=ax)
        ax.set_title("Feature Importance in Crop Recommendation")
        st.pyplot(fig)

if __name__ == "__main__":
    main()

# Import necessary libraries
import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Create a QnA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Streamlit App
st.title("Land Irrigation Recommendation System")

# Sidebar to upload or provide image URL
image_source = st.sidebar.radio("Choose image source:", ('Upload Image', 'URL Image'))

# Load image based on user selection
if image_source == 'Upload Image':
    uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
else:
    image_url = st.sidebar.text_input("Enter image URL")
    if image_url:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

# Display the image
if 'image' in locals():
    st.image(image, caption="Uploaded Image", use_column_width=True)

# Define question input
st.subheader("Ask a Question about the Image")
question = st.text_input("Enter your question", "Does the land need irrigation?")

# Answer the question
if question and 'image' in locals():
    # Use a fixed context for now, but this could be extended in the future
    context = "Need of Irrigation or Not"
    result = qa_pipeline(question=question, context=context)

    # Display result
    st.write(f"Answer: **{result['answer']}**")

    # Function to determine if water is needed
    def needs_water(answer):
        if "water" in answer.lower():
            return "The land does not need water."
        else:
            return "The land needs water."

    # Print if water is needed based on the model's answer
    water_message = needs_water(result['answer'])
    st.write(water_message)
    
    # Display the image with a title showing the answer
    plt.imshow(image)
    plt.axis('off')
    plt.title(f" {result['answer']}")
    st.pyplot(plt)
