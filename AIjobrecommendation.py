import json
import numpy as np
import pandas as pd
import pymongo
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from config import MONGO_URI

# Connect to MongoDB
client = pymongo.MongoClient(MONGO_URI)
db = client['alumni_database']

# Fetch Users & Jobs from MongoDB
users = list(db['profiles'].find({}, {"_id": 0, "name": 1, "skills": 1, "bio": 1, "location": 1, "experience": 1}))
jobs = list(db['jobs'].find({}, {"_id": 0, "title": 1, "required_skills": 1, "experience_required": 1, "company": 1, "location": 1, "description": 1}))

# Prepare Job Dataset for Training
job_texts, job_labels = [], []

for job in jobs:
    skills = job.get("required_skills", "")
    experience_required = str(job.get("experience_required", "0"))
    
    if skills:
        job_texts.append(skills + " " + experience_required)
        job_labels.append(job.get("title", "Unknown"))

# Ensure valid jobs exist before training
if not job_texts:
    raise ValueError("No valid jobs available!")

# Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
X_train = vectorizer.fit_transform(job_texts)

# Encode job titles
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(job_labels)

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Function to recommend jobs and store notifications
def recommend_jobs(user):
    user_skills = " ".join(user.get("skills", []))
    user_bio = user.get("bio", "").strip()
    user_location = user.get("location", "")
    user_experience = user.get("experience", 0)
    
    if not user_skills and not user_bio:
        return None  # Skip users with no valid skills or bio
    
    user_text = user_skills + " " + user_bio
    X_test = vectorizer.transform([user_text])

    # Predict job category
    predicted_label = model.predict(X_test)[0]
    recommended_job = label_encoder.inverse_transform([predicted_label])[0]
    
    for job in jobs:
        job_experience_numeric = int(job.get("experience_required", "0") or 0)
        
        if job["title"] == recommended_job and job_experience_numeric <= user_experience:
            notification = {
                "user_name": user["name"],
                "job_title": job["title"],
                "company": job["company"],
                "location": job["location"],
                "description": job["description"],
                "status": "unread"
            }
            db.notifications.insert_one(notification)
            return job
    return None

# Iterate over users and recommend jobs
for user in users:
    recommended_job = recommend_jobs(user)
    if recommended_job:
        print(f"✅ Job recommendation for {user['name']} stored in notifications!")
        print("\n🎯 Job Recommendation Found!")
        print(f"User: {user['name']}")
        print(f"Job Title: {recommended_job['title']}")
        print(f"Company: {recommended_job['company']}")
        print(f"Location: {recommended_job['location']}")
        print(f"Description: {recommended_job['description']}\n")
    else:
        print(f"❌ No suitable jobs found for {user['name']}.")

# Close MongoDB connection
client.close()

