import altair as alt

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