from streamlit import cache_data
from wordcloud import WordCloud
import dtreeviz

import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np

from Modules.data_utils import prettify_data

#@cache_data(show_spinner=True)
def visualize_decitree(_model, df, target, readable_df=None, cmap="viridis"):
    """
    Visualizes a decision tree, given a model, a training dataset, and a target column.
    You can optionally pass in a readable_df, which is a prettified version of the dataset
    used for training. If you don't pass one in, it will be generated.
    You can also optionally pass in a colormap, which will be used to color the classes.
    """
    df = df.dropna()
    if readable_df is None:
        readable_df = prettify_data(df)
    if isinstance(cmap, str):
        cmap = cm.get_cmap(cmap)
    
    # Setup classes
    strfix = lambda s: s.replace("_ms","").replace("_", " ").title()
    feature_names = list(map(strfix, readable_df.drop(target, axis=1).columns))
    class_names = list(map(strfix, readable_df[target].unique().astype(str)))
    p_target_name = strfix(target)
    
    # Setup colors
    if 'matplotlib.colors' in str(type(cmap)).lower():
        # Draw colors from the colormap
        colors_list = cmap(np.linspace(0.05, 0.8, len(class_names)))
        cmap = [colors.rgb2hex(color) for color in colors_list]
    
    viz = dtreeviz.model(_model, 
                X_train=df.drop(target, axis=1), 
                y_train=df[target],
                target_name=p_target_name,
                feature_names=feature_names, 
                class_names=class_names
                )
    
    return viz.view(colors={'classes': [cmap] * (1+len(class_names))}).svg()

@cache_data(show_spinner=True)
def generate_wordcloud(input, width=800, height=400, colormap='YlGnBu'):
    if type(input) is str:
        w_freq = input.split()
    if type(input) is list:
        w_freq = {word: w_freq.get(word, 0) + 1 for word in input}
    if not 'w_freq' in locals():
        w_freq = input
        #for word in input: w_freq[word] = w_freq.get(word, 0) + 1
    
    wordcloud = WordCloud(
        width=width, height=height,
        background_color=None,
        mode='RGBA',
        colormap=colormap,
        max_words=50,
        prefer_horizontal=0.8,
        min_font_size=10,
        max_font_size=100,
        normalize_plurals=False,
        random_state=42,
        collocations=False,
        font_path="./Fonts/Helvetica.ttf"
    ).generate_from_frequencies(w_freq)
    
    wordcloud.recolor(color_func=None, random_state=42)
    
    # Return something that can easily be written to streamlit
    return wordcloud.to_image()

