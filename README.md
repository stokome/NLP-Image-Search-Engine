# NLP-Image-Search-Engine
NLP-Image-Search-Engine is a project that allows users to search for images based on natural language or image queries. The project uses OpenAI's CLIP (Contrastive Language-Image Pre-Training) model to generate embeddings for both the images in the dataset and the user's query. The embeddings are used to calculate the similarity between the images and the query, and the top-ranked images are returned to the user. The query can be natural language text or image provided by the user. The app is deployed using streamlit.

## Screenshots
![App Screenshot](https://github.com/stokome/NLP-Image-Search-Engine/blob/main/screenshots/img1.png)
![App Screenshot](https://github.com/stokome/NLP-Image-Search-Engine/blob/main/screenshots/img2.png)

# Dataset
The project uses the Unsplash dataset of 25k images, which can be downloaded from the official Unsplash website or github repository.

# Requirements
The project requires the following dependencies to be installed:
1. Python 3.6+
2. Tensorflow
3. Transformers
4. Streamlit
5. numpy
6. pandas
7. requests

# Usage
To use the project, follow these steps:

1. Clone the repository: 
  Use the following code to clone the repository-                                                                                                                         
  $git clone https://github.com/stokome/Nlp-Image-Search-Engine.git
  
2. Install the dependencies:
  Use the following code to install dependencies-                                                                                                                         
  $cd Nlp-Image-Search-Engine                                                                                                                                             
  $pip install -r requirements.txt
  
3. Start the app:                                                                                                                                                         
  $streamlit run app.py
  
