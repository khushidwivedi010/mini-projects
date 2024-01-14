import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from googletrans import Translator                                      
#from sklearn.svm import SVC      
import joblib       
# Load the dataset                                                                                   
df = pd.read_csv('emotions1.csv')    

# Separate features (content) and target (sentiment) 
X = df['content']
y = df['sentiment'] 

# Split the data into training and testing sets    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)    

# Vectorize the text data
vectorizer = TfidfVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)     
joblib.dump(vectorizer, 'vectorizer.pkl')  
X_test_vectorized = vectorizer.transform(X_test)  

# Train the model with modified parameters  
model = LogisticRegression(max_iter=1000, random_state=42) 
model.fit(X_train_vectorized, y_train) 

 

joblib.dump(model, 'sentiment_model.pkl') 
# Evaluate the model
y_pred = model.predict(X_test_vectorized)

#accuracy = accuracy_score(y_test, y_pred)
#print("Accuracy:", accuracy) 

# Function to predict sentiment of a new text
def predict_sentiment(text):    
    text_vectorized = vectorizer.transform([text])   
    sentiment = model.predict(text_vectorized)[0]     
    return sentiment   



# Example usage  
translator = Translator()   
new_text = '''i am very upset about this product i received but it works well'''     
#new_text = '''I should be sleep, but im not! thinking about an old friend who I want. but he's married now. damn, &amp; he wants me '''
translated_text = translator.translate(new_text, dest='en')         
predicted_sentiment = predict_sentiment(translated_text.text)         
print("Predicted sentiment:", predicted_sentiment)          

