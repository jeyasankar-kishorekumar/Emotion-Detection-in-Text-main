# Core Packages
from track_utils import create_page_visited_table, add_page_visited_details, view_all_page_visited_details, add_prediction_details, view_all_prediction_details, create_emotionclf_table
import hashlib
import sqlite3
import streamlit as st
import altair as alt
import plotly.express as px

# EDA Packages
import pandas as pd
import numpy as np
from datetime import datetime

# Load Model
import joblib

pipe_lr = joblib.load(open("./models/emotion_classifier_pipe_lr.pkl", "rb"))

# Track Utils

# Function

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗",
    "joy": "😂", "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

# DB Management
conn = sqlite3.connect('data.db')
c = conn.cursor()

# DB Functions

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS registration_table(username TEXT, password TEXT, first_name TEXT, last_name TEXT, age INT, gender TEXT, email TEXT)')

# Create a new table for user registration data
create_usertable()  # Create the registration table

def add_userdata(username, password, first_name, last_name, age, gender, email):
    c.execute('INSERT INTO registration_table(username, password, first_name, last_name, age, gender, email) VALUES (?,?,?,?,?,?,?)',
              (username, password, first_name, last_name, age, gender, email))
    conn.commit()

def login_user(username, password):
    c.execute('SELECT * FROM registration_table WHERE username = ? AND password = ?',
              (username, password))
    data = c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM registration_table')
    data = c.fetchall()
    return data

def check_user_table():
    c.execute('SELECT * FROM registration_table')
    data = c.fetchall()
    return data

# Validation Functions
def is_valid_email(email):
    return email.endswith('@gmail.com')

def is_valid_name(name):
    return name.isalpha()

def is_valid_password(password):
    return len(password) == 8  # Require the password to be exactly 8 characters long

# Main Application

def main():
    st.title("EMOTION CLASSIFIER APP")
    menu = ["Home", "Registration", "Login", "About", "Logout", "Admin", "Monitor"]
    choice = st.sidebar.selectbox("Menu", menu)
    create_page_visited_table()
    create_emotionclf_table()
    st.session_state.admin_logged_in = False  # Initialize admin login state
    logged_in = False
    username = ""

    if choice == "Login" and not logged_in and not st.session_state.admin_logged_in:
        st.subheader("Login Section")
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            if not is_valid_password(password):
                st.error("Invalid password format. Password should be exactly 8 characters long.")
            else:
                hashed_pswd = make_hashes(password)
                result = login_user(username, check_hashes(password, hashed_pswd))
                if result:
                    st.success("Logged In as {}".format(username))
                    logged_in = True
                    choice = "Home"  # Automatically move to the home page after successful login
                else:
                    st.error("Invalid credentials. Please try again.")
                    st.write("User Table Data:", check_user_table())

    if not logged_in and not st.session_state.admin_logged_in:
        if choice == "Admin":
            admin_username = "admin"
            admin_password = "admin"
            st.subheader("Admin Login Section")
            username = st.text_input("Admin Name")
            password = st.text_input("Admin Password", type='password')

            submit = st.button("Submit")
            cancel = st.button("Cancel")

            if submit:
                if username == admin_username and password == admin_password:
                    st.session_state.admin_logged_in = True
                    st.success("Admin Login Successful")
                    choice = "View User Data"  # Redirect to the "View User Data" page
                else:
                    st.error("Invalid admin credentials. Please try again.")

            if cancel:
                choice = "Home"

        if choice == "Monitor":
            add_page_visited_details("Monitor", datetime.now())
            st.subheader("Monitor App")

            with st.expander("Page Metrics"):
                page_visited_details = pd.DataFrame(view_all_page_visited_details(), columns=[
                    'Page Name', 'Time of Visit'])
                st.dataframe(page_visited_details)

                pg_count = page_visited_details['Page Name'].value_counts().rename_axis('Page Name').reset_index(name='Counts')
                c = alt.Chart(pg_count).mark_bar().encode(
                    x='Page Name', y='Counts', color='Page Name')
                st.altair_chart(c, use_container_width=True)

                p = px.pie(pg_count, values='Counts', names='Page Name')
                st.plotly_chart(p, use_container_width=True)

    if choice == "About":
        add_page_visited_details("About", datetime.now())
        st.write("Welcome to the Emotion Detection in Text App! This application utilizes the power of natural language processing and machine learning to analyze and identify emotions in textual data.")

        st.subheader("Our Mission")
        st.write("At Emotion Detection in Text, our mission is to provide a user-friendly and efficient tool that helps individuals and organizations understand the emotional content hidden within text. We believe that emotions play a crucial role in communication, and by uncovering these emotions, we can gain valuable insights into the underlying sentiments and attitudes expressed in written text.")

        st.subheader("How It Works")
        st.write("When you input text into the app, our system processes it and applies advanced natural language processing algorithms to extract meaningful features from the text. These features are then fed into the trained model, which predicts the emotions associated with the input text. The app displays the detected emotions, along with a confidence score, providing you with valuable insights into the emotional content of your text.")

        st.subheader("Key Features:")

        st.markdown("##### 1. Real-time Emotion Detection")
        st.write("Our app offers real-time emotion detection, allowing you to instantly analyze the emotions expressed in any given text. Whether you're analyzing customer feedback, social media posts, or any other form of text, our app provides you with immediate insights into the emotions underlying the text.")

        st.markdown("##### 2. Confidence Score")
        st.write("Alongside the detected emotions, our app provides a confidence score, indicating the model's certainty in its predictions. This score helps you gauge the reliability of the emotion detection results and make more informed decisions based on the analysis.")

        st.markdown("##### 3. User-friendly Interface")
        st.write("We've designed our app with simplicity and usability in mind. The intuitive user interface allows you to effortlessly input text, view the results, and interpret the emotions detected. Whether you're a seasoned data scientist or someone with limited technical expertise, our app is accessible to all.")

        st.subheader("Applications")

        st.markdown("The Emotion Detection in Text App has a wide range of applications across various industries and domains. Some common use cases include:")
        st.write("- Social media sentiment analysis")
        st.write("- Customer feedback analysis")
        st.write("- Market research and consumer insights")
        st.write("- Brand monitoring and reputation management")
        st.write("- Content analysis and recommendation systems")

    if logged_in and choice == "Home":
        add_page_visited_details("Home", datetime.now())
        st.subheader("Emotion Detection in Text data")
        with st.form(key='emotion_clf_form'):
            raw_text = st.text_area("Type Here")
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1, col2 = st.columns(2)
            # Apply Function Here
            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            add_prediction_details(raw_text, prediction, np.max(
                probability), datetime.now())

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write("{}:{}".format(prediction, emoji_icon))
                st.write("Confidence: {}".format(np.max(probability)))

            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(
                x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

    elif choice == "Logout":
        logged_in = False
        st.success("Logged Out")
        st.info("You are now logged out. Please log in again to access the application.")

    if choice == "Registration":
        # Registration Form
        st.title("Registration Form")

        # Input fields
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        reg_username = st.text_input("Username (Use for Login)")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        gender = st.radio("Gender", ("Male", "Female", "Other"))
        email = st.text_input("Email (Gmail)")
        reg_password = st.text_input("Password (Use for Login)", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        # Validation and submission
        if st.button("Register"):
            if not is_valid_email(email):
                st.error("Invalid email format. Please use a Gmail address.")
            elif not is_valid_name(first_name):
                st.error("Invalid first name. Use only alphabets.")
            elif not is_valid_name(last_name):
                st.error("Invalid last name. Use only alphabets.")
            elif len(reg_password) != 8:  # Ensure the password is exactly 8 characters long
                st.error("Password must be exactly 8 characters long.")
            elif reg_password != confirm_password:
                st.error("Passwords do not match. Please re-enter.")
            else:
                create_usertable()  # Create the registration table
                hashed_pswd = make_hashes(reg_password)
                add_userdata(reg_username, hashed_pswd, first_name, last_name, age, gender, email)
                st.success("Registration successful! You can now log in.")
                choice = "Login"  # Redirect to the login page after successful registration

    if st.session_state.admin_logged_in:
        if choice == "View User Data":
            add_page_visited_details("View User Data", datetime.now())
            st.subheader("View User Registration Data")

            # Fetch user registration data from the database
            user_data = view_all_users()

            if user_data:
                # Display user registration data in a DataFrame
                user_df = pd.DataFrame(user_data, columns=[
                    "Username", "Password", "First Name", "Last Name", "Age", "Gender", "Email"])
                st.dataframe(user_df)
            else:
                st.warning("No user registration data found in the database.")

if __name__ == '__main__':
    main()
