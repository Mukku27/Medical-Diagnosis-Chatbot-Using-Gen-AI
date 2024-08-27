import pandas as pd
import re

# Load the dataset
df = pd.read_csv('Dataset/Raw Data/mtsamples.csv')

def extract_symptoms(text):
    # Check if text is a string, otherwise return a default value
    if not isinstance(text, str):
        return 'No specific symptoms found'

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

# Apply the function to extract symptoms from the 'transcription' column
df['symptoms'] = df['transcription'].apply(extract_symptoms)

# Save the processed data to a new CSV file
df.to_csv('Dataset/Preprocessed Data/processed_dataset.csv', index=False)
