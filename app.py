import pickle
import streamlit as st

model = pickle.load(open("spam.pkl","rb"))
cv = pickle.load(open("vectorizer.pkl","rb"))

def main():
    st.title("Email Spam Classifier")
    st.subheader("It is used to classify whether any text is spam or not spam")
    msg = st.text_input("Enter the text : ")
    if st.button("Predict"):
        data = [msg]
        vect = cv.transform(data).toarray()
        prediction = model.predict(vect)
        result = prediction[0]

        if result == 1:
            st.error("This text is spam")
        else:
            st.success("This text is not spam")

main()
