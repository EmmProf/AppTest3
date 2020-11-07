# NameFeaturesDetector

The aim of this project is: given a name's morphology predict features of this name like its locality (where people with this name live, not origin), gender, 
and if it is a first or a last name.  
The model used to extract features was a multi-label, multi-tasks CNN model (inspired by inception networks architecture).  
From the last layer embeddings we derive a similarity measure between names.  
You can test the model using the [app] (https://name-features-detector.herokuapp.com/)
