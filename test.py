import streamlit as st
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from dtreeviz.trees import dtreeviz
import base64

# Load iris dataset
iris = load_iris()

# Create decision tree classifier
clf = DecisionTreeClassifier()

# Fit the classifier to the data
clf.fit(iris.data, iris.target)

# Create a Streamlit app
st.set_page_config(page_title="Interactive Decision Tree", page_icon=":deciduous_tree:")
st.header("Interactive Decision Tree")
st.subheader("Iris Dataset")

# Add a slider to select the maximum depth of the tree
max_depth = st.slider("Maximum Tree Depth", min_value=1, max_value=10, value=3)

# Generate the decision tree using dtreeviz
viz = dtreeviz(clf, iris.data, iris.target,
               target_name='species',
               feature_names=iris.feature_names,
               class_names=list(iris.target_names),
               fancy=False)

# Convert the SVG output to a string
svg_string = viz.svg()

# Create a unique ID for the SVG element
svg_id = "my-svg"

# Create a base64 encoded version of the SVG string
svg_base64 = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')

# Create the HTML for the SVG element
svg_html = f"""
    <div style='overflow: scroll; width: 100%; height: 800px;'>
        <object data="data:image/svg+xml;base64,{svg_base64}" type="image/svg+xml" id="{svg_id}">
            {svg_string}
        </object>
    </div>
"""

# Render the SVG element using st.components.v1.html
st.components.v1.html(svg_html)


# Add an explanation of how to interact with the tree
st.markdown("Click on a node to see the selected node ID.")
