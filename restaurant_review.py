import pandas as pd
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import joblib
import os
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

DATASET = "Yelp/yelp_review_full"
MODEL_FILE = "sentiment_model.pkl"


#function 1: data pre-processing: load yelp dataset and process raw data
def load_and_preprocess_data():
    try:
        st.info("Loading Yelp dataset...")
        dataset = load_dataset(DATASET)
        #clean 1: get text and labels from dataset
        train_data = pd.DataFrame({
            'review': dataset['train']['text'],  
            'label': dataset['train']['label']
        })
        test_data = pd.DataFrame({
            'review': dataset['test']['text'],
            'label': dataset['test']['label']
        })

        #clean 2: cast all reviews to strings
        train_data['review'] = train_data['review'].astype(str).str.strip()
        test_data['review'] = test_data['review'].astype(str).str.strip()
        #clean 3: drop rows with empty reviews or column value
        train_data = train_data[train_data['review'].str.len() > 0]
        test_data = test_data[test_data['review'].str.len() > 0]

        
        st.success("Data load and clean OK")
        return train_data, test_data
    
    except Exception as e:
        st.error(f"error: {e}")
        return None, None


#function 2: map labels to good, neutral, bad.  labels range from 1-5
def map_labels(df):
    def labelmap(x):
        if x==1 or x==2:
            return "bad"
        elif x == 3:
            return "neutral"
        else:
            return "good"
    
    df['label']=df['label'].apply(labelmap)
    return df

#function 3: train sentiment-analysis model using Naive Bayes
def train_model(train_data, test_data):
    try:
        st.info("Training the model...")
        #step 1: map labels to two splitted datasets
        train_data = map_labels(train_data)
        test_data = map_labels(test_data)
        
        #clean 4: cast all reviews to strings and drop rows with missing row
        train_data['review'] = train_data['review'].astype(str).str.strip()
        test_data['review'] = test_data['review'].astype(str).str.strip()
        train_data = train_data[train_data['review'].str.len() > 0]
        test_data = test_data[test_data['review'].str.len() > 0]
        
        #step 2: split both datasets into features and labels
        X_train = train_data['review']
        y_train =train_data['label']
        X_test= test_data['review'] 
        y_test = test_data['label']
        #step3: combine steps for data flow; process out stopwords
        pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english', min_df=2)), ('clf', MultinomialNB()),])
        #step 4: train the model
        pipeline.fit(X_train, y_train)
        
        #step 5: evaluate the model
        y_predict = pipeline.predict(X_test)
        report = classification_report(y_test, y_predict)
        st.text("Model Eval:")
        st.code(report)
        
        #save model
        joblib.dump(pipeline, MODEL_FILE)
        st.success("Model trained successfully!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    
    
    
    
    except Exception as e:
        st.error(f"Error training the model: {e}")



#function 4: predict new input
def predict_sentiment(review):
    try:
        if not os.path.exists(MODEL_FILE):
            st.error("Error: model file not found")
            return None
        
        pipeline = joblib.load(MODEL_FILE) #load the model
        prediction = pipeline.predict([review])[0]
        return prediction
    except Exception as e:
        st.error(f"Error in prediction code: {e}")
        return None

#interface
def main():
    st.title("Restaurant Review Sentiment Analysis")
    st.write("### Classify reviews as Good, Neutral, or Bad")
    
    # Sidebar Navigation
    option = st.sidebar.selectbox("Menu", ["Train Model", "Classify Review"])
    
    if option == "Train Model":
        # Load Dataset and Train Model
        train_data, test_data = load_and_preprocess_data()
        if train_data is not None and test_data is not None:
            train_model(train_data, test_data)
    
    elif option == "Classify Review":
        #new review textbox
        st.write("#### Enter a review to classify:")
        review = st.text_area("Review", "Type your review here...")
        
        if st.button("Classify Sentiment"):
            if review.strip():
                sentiment = predict_sentiment(review)
                if sentiment:
                    st.success(f"Predicted Sentiment: **{sentiment.capitalize()}**")
            else:
                st.warning("Please enter a review to classify.")

#run
if __name__ == "__main__":
    main()
