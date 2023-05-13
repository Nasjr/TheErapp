# TheErapp: Analyze Rap Text and Compare Similarities Between Songs of Different Artists
TheErapp is an app that analyzes rap text and compares similarities between songs of different artists using tf-idf and cosine similarity. It also shows different text mining data, such as the number of words and average words of an artist, and visualizes the artist's number of words over time and their lyrics in a word cloud. The app is built using Streamlit and is deployed to Render.
---------------------
## 🚀 Getting Started
To run TheErapp locally, you'll need to have Python 3.7 or higher installed on your system. You can then install the required Python packages using the following command:

basic
Copy
pip install -r requirements.txt
Once the packages are installed, you can start the app by running the following command:

Copy
streamlit run app.py
The app should open in your default web browser.
--------------------- 
## Usage 📖 

When you first open the app, you'll see a dropdown menu where you can select the name of an artist whose lyrics you want to analyze. After selecting the artist name and clicking the "Analyze" button, the app will retrieve the lyrics of the artist's songs using the Genius API. The app will then perform text mining on the lyrics to extract different features, such as the number of words and average words per song.

You can then use the different tabs in the app to visualize the data and compare it with other artists. The "Similarity" tab allows you to compare the similarity between the lyrics of different artists using tf-idf and cosine similarity. The "Word Cloud" tab displays a word cloud of the artist's lyrics, where the size of each word represents its frequency in the lyrics.

## Deployment

TheErapp is deployed to Render. To deploy the app to Render, you'll need to create an account on Render and follow the deployment instructions in the Render documentation.

## Credits

TheErapp was created by [Mahmoud Nasser]. The app uses the [Genius API](https://docs.genius.com/) to retrieve lyrics and the [Streamlit](https://streamlit.io/) library for the web interface. The app also uses the [scikit-learn](https://scikit-learn.org/) library for text mining and the [WordCloud](https://github.com/amueller/word_cloud) library for generating word clouds.

## License

TheErapp is released under the [MIT License](https://opensource.org/licenses/MIT).
