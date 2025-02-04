import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



# **Introduction**

st.title("🎵 EDM Genre Analysis & Classification")
st.image("cover_image.png", use_container_width=True)

st.markdown("### The Evolution and Classification of EDM")

st.write("""Electronic Dance Music (EDM) started in small, hidden clubs and has grown into a worldwide sensation, with countless styles—from the steady beats of Techno to the joyful rhythms of House, the gritty bass of Dubstep, and the soaring melodies of Trance. But here's the big question: How do these genres actually sound different from each other?""")
st.write("""You know how a Techno track feels like a futuristic machine pumping energy, while House makes you want to dance with its warm, soulful vibes? Or why Dubstep hits you like a wall of sound, but Trance feels like floating on clouds? Musicians might say, “It's all about the feeling,” but we wanted to dig deeper. Instead of just guessing, we're using science and technology to find answers.""")
st.write("""Think of it like this: Imagine taking apart a song the way you would take apart a clock. We are looking at the 'pieces' of sound—like how high or low the notes are, how the energy changes over time, and how the layers of sound fit together. For example, maybe Dubstep sounds aggressive because it has wild jumps between deep bass and sharp highs, while Trance stays smooth with long, glowing melodies. By measuring these details, we can finally say, “Here is exactly what makes House'House' and Dubstep 'Dubstep'!""")
st.write("This story isn\'t just for scientists—it's for anyone who loves music. By understanding the “recipe” of a genre, producers can create better tracks, fans can discover music they'll love, and we can all appreciate the magic behind the beats. Let's break it down together—no fancy words, just the cool science behind the music we adore.")



# **Genre Distribution**

st.header("🎶 Genre Distribution")

df = pd.read_csv("train_sample.csv")
fig = px.bar(df["label"].value_counts(), labels={'index': 'Genre', 'value': 'Count'}, title="Genre Count")
st.plotly_chart(fig)

st.caption("What are those? To me, these are just a bunch of bars that look almost the same height.")
st.write("Here, we have a bar chart showdown of EDM genres, where each genre is flexing its musical muscles to prove who dominates the dataset! Think of it like a popularity contest at a massive festival—except instead of fans, we’re counting tracks.Looking at the chart, it’s clear that no single genre is stealing the spotlight—most genres have roughly the same number of tracks. This means we’ve got a balanced dataset, so every genre gets a fair shot at being analyzed!")
st.caption("Sounds interesting. But what if some of them is much taller than others?")
st.write("If one genre had way more tracks than the others, our analysis might be biased, like having a DJ that only plays House music all night (that would be so boring). Since the dataset is evenly spread, our machine learning models can learn fairly from all genres—no favoritism here! If your favorite genre is at the bottom of the chart, don’t worry—it’s about quality, not quantity.")
st.caption("Well, as an Ambient fan I'm glad to hear this.")
st.write("What’s Next? Now that we know how our dataset is structured, let’s dig deeper—what makes these genres sound different? Stay tuned for the spectral feature showdown!")


# **Feature Analysis**

st.header("📊 Spectral Features by Genre")
st.write("If EDM were a high-tech cooking show, then spectral features would be the ingredients that define the flavor of each genre. Some genres love spicy high frequencies (like Techno), while others prefer smooth and creamy bass (like Lofi). Just like a chef carefully selects ingredients, producers tweak these spectral properties to craft the perfect sonic experience.")

st.write("Each genre has its signature sound. For example:")
st.write("House & Techno has higher spectral centroid which determines brightness (like how much spice in a dish).")
st.write("Lofi & Ambient has lower rolloff which indicates energy in low frequencies (like adding extra butter for richness).")
st.write("Dubstep & Drum & Bass has wider spectral bandwidth which measures spread of frequencies (like how complex the flavors are).")



selected_feature = st.selectbox("Select a feature to compare:", df.columns[:-2])
fig = px.box(df, x="label", y=selected_feature, title=f"{selected_feature} by Genre")
st.plotly_chart(fig)

st.caption("I'm only seeing a bunch of boxes, please explain")
st.write("Think of this box plot as a musical talent show—each genre steps onto the stage, showing off how its spectral feature varies. Some genres are consistent performers, while others have wildly experimental tracks that don’t fit the norm!")
("Here's how to read the stage performance:")
("""The box represents the middle 50% of values—this is where most tracks in a genre hang out. The line inside the box is the median—the "average performer" for that genre. The whiskers show the full range of normal values—stretching their arms out to show flexibility. Dots outside the whiskers? These are the rebels—tracks that push genre boundaries and break expectations!""")
("Taking Root Mean Square Energy (RMSE) as an example, it measures the loudness fluctuations within a track. A higher RMSE generally indicates a track with punchy, dynamic energy, while a lower RMSE suggests a softer, more consistent sound. If we click on the bar and take a closer look at the selected spectral feature rmse_mean and see how different genres perform, we can find out that Phonk is like that contestant who can do everything from opera to rap—it has a huge range of values, meaning tracks in this genre vary a lot in spectral characteristics. Lofi & Psytrance are the disciplined ones—their values are tightly packed, meaning these genres have a consistent sound that artists stick to. Dubstep & Ambient have some rule-breakers—those outliers could be experimental tracks or hybrids that mix with other genres.")
("Due to space limitations, we can't walk you through every selectable feature from the dataset, but feel free to explore them all on your own!")


st.caption("So, what's the conclusion? Do we get anything that the data didn't explicitly tell us?")
st.write("The most important conclusion is: genres are not mololithic."
""" Even within a single genre, spectral properties vary widely, and some tracks blur the boundaries between styles."""
""" If you were a producer and you want to make your track "more techno", you can tweak spectral centriod, MFCCs and RMSE to match the genre's signature.""")
st.write("Now that we've learned individual features, what about putting them together? Can we cluster genres based on their spectral properties? Let's move on to find the answer.")





# **Clustering**

st.header("🌍 Clustering of EDM Genres")

st.write("To understand the relationships between EDM genres, we need a way to visualize high-dimensional data in a meaningful way." 
"Songs have many spectral features, but humans can only intuitively interpret 2D or 3D graphs." 
"This is where dimensionality reduction techniques like PCA, UMAP, and t-SNE come in.")

st.caption("But why clustering?")
st.write("Different from other music genres, EDM genres are highly overlap and blend (e.g. Tech House is a mix of Techno and House). By applying PCA, UMAP and t-SNE, we can see: 1. Which genres are naturally grouped together. 2. Which genres are distinct and separate. 3. Which genres have significant overlap, indicating subgenres.")


dim_reduction = st.selectbox("Select Dimensionality Reduction (You can try all three of them):", ["PCA", "UMAP", "t-SNE"])

features = df.drop(columns=["label"])
if dim_reduction == "PCA":
    reducer = PCA(n_components=2)

elif dim_reduction == "UMAP":
    reducer = UMAP(n_components=2, random_state=42)
elif dim_reduction == "t-SNE":
    reducer = TSNE(n_components=2, random_state=42)

#This is the chart, DO NOT change this
projection = reducer.fit_transform(features)
df["Dim1"], df["Dim2"] = projection[:, 0], projection[:, 1]
fig = px.scatter(df, x="Dim1", y="Dim2", color="label", title=f"{dim_reduction} Projection of EDM Genres")
st.plotly_chart(fig)

#This feature allows the related text to appear after the user selected the chart they wants to look at.
if dim_reduction == "PCA":
    st.caption("I selected PCA and it looks like a bag of Skittles just exploded on my screen.")
    st.write("""This is actually a Principal Component Analysis projection, which is a fancy way of saying "Let’s take our super high-dimensional EDM data and squish it down to just two dimensions, so our human brains (and eyes) can actually understand it." """
    """Each dot represents a song, and its position is determined by its compressed spectral features. Similar songs should, in theory, cluster together—but as you can see, real-world music isn’t always so neatly categorized.""")
    st.write(""" Those green blob in the bottem left are Ambient, which tends to have a distinct, tightly packed sound signature."""
    """ Lofi (blue dots) also forms a cluster on the left, suggesting its tracks share a strong sonic identity.""")

elif dim_reduction == "UMAP":
    st.caption("Wow, that's better than a bag of exploded Skittles. It looks like a space map.")
    st.write("UMAP (Uniform Manifold Approximation and Projection) is an algorithm that’s really good at preserving local relationships. Unlike PCA, which tries to retain overall variance, UMAP groups similar songs more tightly together—and as you can see, it does a way better job at forming actual clusters!")
    st.caption("I noticed that some genres are still all over the place. Is this normal?")
    st.write("You are right! Trap, House/Big Room House and Phonk can be spotted in many places on this chart rather than staying in a specific cluster. This means that their spectral properties are more fluid and overlapping."
    " Real-world music isn’t neatly categorized. Genres evolve, blend, and defy strict classification.")


elif dim_reduction == "t-SNE":
    st.caption("I know this one, it's a cartoon dinosaur's screaming face. Those green dots form the tongue and the blue & purple dots are the jaw. I can also see a big nose on the right. ")
    st.write("t-SNE (t-Distributed Stochastic Neighbor Embedding) is an algorithm designed to emphasize local relationships—meaning it tries really, really hard to group similar tracks together while pushing dissimilar tracks apart. ")
    st.write(" The result? I would say it is a twisted structure that looks like a neon-lit jungle gym."
    " Genres are clumped in some areas, but others still blend chaotically, because EDM is one big happy genre-mixing family."
    " Those blue & green clusters on the bottom left (which is what you are referring to tongue and jaw) are actually Ambient and Lofi. They both are considered as atmospheric."
    """ The big "nose" in the middle is the hybrid EDM zone, formed mainly by DnB, Trance and Dubstep.""")



# **MLGC**

st.header("🤖 Train a Genre Classification Model")
st.write("So far, we've visualized our EDM genres using dimensionality reduction techniques like PCA. But here’s the big question: 👉 Can we teach a machine to recognize these genres automatically?")
st.caption("I have a better question: why do we need to teach the machine?")

st.write(" Instead of just looking at those pretty clusters, models allow us to actually predict the genre of an unknown song based on its spectral characteristics. This model will learn from existing data and try its best to label new tracks accurately. And once we've got our model trained, we can push it even further—by identifying tracks that don’t fit neatly into any category. These outliers could be experimental songs, hybrids, or misclassifications, offering insights into how genres evolve and intersect.""")



if "label_encoded" not in df.columns:
    df["label_encoded"] = df["label"].astype("category").cat.codes  # Convert genres to numeric values


drop_columns = ["label"]  
if "Dim1" in df.columns and "Dim2" in df.columns:
    drop_columns.extend(["Dim1", "Dim2"])  # Drop only if they exist

X = df.drop(columns=drop_columns)
y = df["label_encoded"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# This is the button to train model
if st.button("Click This To Train Random Forest Model"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

   
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=df["label"].astype("category").cat.categories)

    # Display results
    st.subheader("✅ Model Accuracy")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.text(report)


# **Impostors**

st.header("🤯 Detecting Imposters")
thresh = st.slider("Set Outlier Threshold", 0.1, 2.0, 1.0, 0.1)


if "model" in globals():
    prediction_confidence = model.predict_proba(X_test).max(axis=1)
    
    # Create a copy of df with only test samples to avoid index mismatch
    df_test = df.iloc[X_test.index].copy()
    df_test["prediction_confidence"] = prediction_confidence
    
    # Find potential imposters
    outliers = df_test[df_test["prediction_confidence"] < thresh]

    fig = px.scatter(outliers, x="Dim1", y="Dim2", color="label", title="Potential Genre Imposters")
    st.plotly_chart(fig)
else:
    st.warning("⚠️ Train the model first before detecting imposters.")

if "model" in globals():
    st.write("You should be seeing a chart that looks like the PCA result, but with less Skittles. ")
    st.write("In this chart, we're detecting imposters—tracks that don’t quite fit into their assigned genre. Think of it like a costume party where everyone is dressed as their favorite genre, but some guests clearly didn’t read the dress code.")
    st.write("Each dot represents a track, and their positions are determined by their spectral characteristics. Most dots cluster together in genre-based groups, but some tracks are way off the expected region. These misfits, the ones that stray too far from their genre’s usual neighborhood, are flagged as potential outliers (imposters).")
    st.write("Why are they imposters?"
    " Some producers love bending the rules. A lofi track with unusually high energy or a hardcore track with softer elements might show up here."
    " Some tracks sit between genres, like a house track with techno influences or a dubstep track with trap beats. These confuse the model because they share features with multiple genres."
    """ Most importantly, the model isn't perfect. Sometimes, a song that belongs to one genre gets misclassified into another, making it an imposter in its assigned group.""")
    st.write("The presence of imposters reminds us that genres aren’t rigid categories. They evolve, blend, and borrow from each other. What seems like an imposter today might define a new subgenre tomorrow!")


# **Conclusion**
st.header("In All Words...What Did We Learn?")
st.write("""
EDM has always been a genre in motion—constantly evolving, blending influences, and defying rigid classification.
Throughout this data story, we’ve unraveled the patterns hidden in sound, from spectral fingerprints that shape a genre’s identity to machine learning models that attempt to categorize (and sometimes misjudge) the fluid nature of music. 
Yet, rather than forcing music into fixed boundaries, what we’ve really uncovered is the complexity of creative expression—how genres overlap, how outliers emerge, and how even the most structured beats can break the mold.
Producers can use these insights to refine their sound, understanding how spectral properties define energy, warmth, or aggression. Listeners, too, can explore EDM in new ways, tracing the connections between their favorite styles and uncovering unexpected similarities. More broadly, as AI and data-driven tools become increasingly integrated into music production, studies like this can help shape how we define, recommend, and even create music in the future.
But perhaps the most important takeaway is that music will always resist being neatly categorized. Just like the anomalies in our visualizations, innovation thrives in the unexpected—those tracks that blur genres, challenge norms, and redefine what EDM can be.                           
""")
