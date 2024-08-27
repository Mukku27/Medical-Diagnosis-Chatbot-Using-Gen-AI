import pandas as pd
import re

# Load the dataset
df = pd.read_csv('Dataset/mtsamples.csv')

# Function to extract symptoms or relevant medical information from the transcription and description
def extract_symptoms(text):
    # Lowercase the text to standardize
    text = text.lower()
    
    # Define some common phrases that might indicate symptoms or key medical information
    symptom_indicators = [
        'symptoms', 'complaints', 'history of present illness', 'patient presents with',
        'subjective', 'objective', 'assessment', 'diagnosis', 'chief complaint'
    ]
    
    # Search for these phrases in the transcription
    symptoms = []
    for indicator in symptom_indicators:
        if indicator in text:
            # Extract the sentence/phrase following the indicator
            match = re.search(rf'{indicator}[:\s]*([^.]*)', text)
            if match:
                symptoms.append(match.group(1).strip())
    
    # Return the extracted symptoms or medical information
    return ' | '.join(symptoms) if symptoms else 'No specific symptoms found'

# Apply the function to the 'transcription' column
df['symptoms'] = df['transcription'].apply(extract_symptoms)

# Save the preprocessed dataset to a new CSV file
df.to_csv('preprocessed_medical_dataset.csv', index=False)

# Display the first few rows of the updated DataFrame
print(df.head())
