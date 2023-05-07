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
               fancy=True)

# Convert the SVG output to a string
svg_string = viz.svg()

# Hacky SVG overrides
import re
svg_string = svg_string.replace(" ", f"""
    
        id="figure" style="width: 300%; height: auto; position: absolute;"
        
""", 1)

# Create the HTML for the SVG element
svg_html = f"""
    <div id="container" style='overflow: none; width: 100%; height: 800px; position: fixed; left: -50%;'>
        {svg_string}
    </div>
    <script>
""" + """
window.onload = function() {
  draggable(document.getElementById('figure'));
}

function draggable(el) {
  el.addEventListener('mousedown', function(e) {
    var offsetX = e.clientX - parseInt(window.getComputedStyle(this).left);
    var offsetY = e.clientY - parseInt(window.getComputedStyle(this).top);
    
    function mouseMoveHandler(e) {
        el.classList.add('active');
      el.style.top = (e.clientY - offsetY) + 'px';
      el.style.left = (e.clientX - offsetX) + 'px';
    }

    function reset() {
      el.classList.remove('active');
      window.removeEventListener('mousemove', mouseMoveHandler);
      window.removeEventListener('mouseup', reset);
    }

    window.addEventListener('mousemove', mouseMoveHandler);
    window.addEventListener('mouseup', reset);
  });
}

</script>
"""

# Render the SVG element using st.components.v1.html
st.components.v1.html(svg_html, height=800, scrolling=True)

# Add an explanation of how to interact with the tree
st.markdown("Click on a node to see the selected node ID.")

# Try with utils
from util import render_draggable
svg_string = viz.svg()
gradient_div = "<div style='background: linear-gradient(to right, #ff0000, #0000ff); width: 200%; height: 200px;'></div>"
render_draggable(gradient_div, 1.5)