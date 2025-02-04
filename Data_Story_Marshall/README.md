Please make sure all the required environment are installed before running.

You can install them by running this command:
pip install streamlit pandas plotly matplotlib seaborn scikit-learn umap-learn

You can click the run.bat to open the file conveniently.

Selected Dataset: EDM Music Genres https://www.kaggle.com/datasets/sivadithiyan/edm-music-genres

There are two files in the dataset: test_data_final.csv and train_data_final.csv
The train_sample.csv is a smaller dataset that have around 10% of the data randomly chosen from the original dataset.
This is to prevent program from crashing (also prevent CPU from burning) because the full dataset is way too big.
Still, for those who have a powerful computer, you can use the original dataset (by changing the source .csv file at line 33). The program will still run properly. 