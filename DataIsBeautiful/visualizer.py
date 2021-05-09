__version__ = '1.0'
__author__ = 'Tyler Ho'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

csgo = pd.read_csv("csgo-matchmaking-damage/mm_master_demos.csv")
csgo.info()
cache = csgo[csgo["map"] == "de_cache"]
cache["dmg"] = cache["hp_dmg"] + cache["arm_dmg"]
cache.info()

cache_img = Image.open("csgo-matchmaking-damage/de_cache.png")

import plotly.graph_objects as go
import plotly.express as px

def add_cache_map(fig):
    fig.add_layout_image(
        dict(
            source=cache_img,
            xref="x",
            yref="y",
            x=-2031,
            y=3187,
            sizex=5783,
            sizey=5427,
            sizing="stretch",
            opacity=0.4,
            layer="above"
        )
    )
    return fig

def add_labels(fig, figtitle, xaxis, yaxis):
    fig.update_layout(
        title=figtitle,
        xaxis_title=xaxis,
        yaxis_title=yaxis,
        font=dict(
            family="Courier New, monospace",
            size=18,
            color="#000000"
        )
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig

dmg_taken_fig = go.Figure()
dmg_taken_fig = px.density_heatmap(cache, x='vic_pos_x', y='vic_pos_y', z="dmg")
dmg_taken_fig = add_cache_map(dmg_taken_fig)
dmg_taken_fig = add_labels(dmg_taken_fig, "Density of Location of Damage Taken on de_cache","","")

dmg_taken_fig.show()

dmg_given_fig = go.Figure()
dmg_given_fig = px.density_heatmap(cache, x='att_pos_x', y='att_pos_y', z="dmg")
dmg_given_fig = add_cache_map(dmg_taken_fig)
dmg_given_fig = add_labels(dmg_taken_fig, "Density of Location of Damage Given on de_cache","","")

dmg_given_fig.show()