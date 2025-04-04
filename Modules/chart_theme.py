import altair as alt
import seaborn as sns

def generate_vega_theme():
    return {
        "config": {
            "background": "white",
            "title": {
                "color": "black",
                "font": "Helvetica",
                "fontSize": 14,
                "fontWeight": "bold"
            },
            "axis": {
                "labelColor": "black",
                "titleColor": "black",
                "gridColor": "lightgray",
                "gridOpacity": 0.2,
                "grid": True
            },
            "header": {
                "labelColor": "black",
                "titleColor": "black",
                "titleFont": "Helvetica",
                "titleFontSize": 16,
                "titleFontWeight": "bold",
                "labelFont": "Helvetica",
                "labelFontSize": 12
            },
            "legend": {
                "labelColor": "black",
                "titleColor": "black",
                "titleFont": "Helvetica",
                "titleFontSize": 12,
                "titleFontWeight": "bold",
                "labelFont": "Helvetica",
                "labelFontSize": 10
            },
            "view": {
                "stroke": "transparent",
                "padding": {'left': 20, 'right': 20, 'top': 60, 'bottom': 5},
                "continuousHeight": 400,
                "continuousWidth": 600,
            }
        }
    }

# Apply the theme
alt.themes.register('cs', generate_vega_theme)
alt.themes.enable('cs')

# Apply all other wanted themes
alt.themes.enable('cs')
sns.set_theme(style="whitegrid")