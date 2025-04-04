import json
from math import sqrt
import os
import argparse
from pathlib import Path
import plotly.graph_objects as go
import numpy as np

# script for plotting the result of a run, e.g: python utils/custom_plotter.py results/xxxxxx-xxxxxx
# --s allows you to prevent saving it to html
# --save_dir allows you to set a custom save directory

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

    # eucledian distance
def dist(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# finds and returns an array of the pareto optimal bids
def findParetoFront(points):
    points = np.array(points)
    paretoFront = np.ones(points.shape[0], dtype=bool)
    for i, point in enumerate(points):
        if paretoFront[i]:
            paretoFront[paretoFront] = np.any(points[paretoFront] > point, axis=1)
            paretoFront[i] = True  
    return paretoFront

    # Method for getting closest point in a list of point compared to the first argument
def closestPoint(point, point_list):
    return min(point_list, key=lambda p: dist(point, p))

def plot(data_dir: str, saveFig: bool, save_dir: str = "plots"):
    session_path = os.path.join(data_dir, 'session_results_trace.json')
    session_data = load_json(session_path)

    actions = session_data.get("actions", [])
    party_profiles = session_data.get("partyprofiles", {})
    agent_names = ["", ""] # some default values

    # load agent json keys
    if party_profiles:
        agent_names = list(party_profiles.keys())

    bids_agent1 = []
    bids_agent2 = []

    #load all bids according to the agent performing it
    for act in actions:
        if "Offer" in act:
            actor = act["Offer"].get("actor")
            utilities = act["Offer"].get("utilities", {})
            if len(utilities) == 2:
                u1, u2 = list(utilities.values())
                if actor == agent_names[0]:
                    bids_agent1.append([u1, u2])
                elif actor == agent_names[1]:
                    bids_agent2.append([u1, u2])

    all_utils = np.array(bids_agent1 + bids_agent2)
    fig = go.Figure()

    agent1_utils = np.array(bids_agent1)
    fig.add_trace(go.Scatter(
        x=agent1_utils[:, 0],
        y=agent1_utils[:, 1],
        mode='markers',
        marker=dict(size=6, color='blue', opacity=0.5),
        name=f'{agent_names[0]} Bids'
    ))

    agent2_utils = np.array(bids_agent2)
    fig.add_trace(go.Scatter(
        x=agent2_utils[:, 0],
        y=agent2_utils[:, 1],
        mode='markers',
        marker=dict(size=6, color='green', opacity=0.5),
        name=f'{agent_names[1]} Bids'
    ))

    # Pareto front
    paretoIndexMask = findParetoFront(all_utils)
    paretoPoints = all_utils[paretoIndexMask]
    paretoPoints = paretoPoints[np.argsort(paretoPoints[:, 0])]

    fig.add_trace(go.Scatter(
        x=paretoPoints[:, 0], y=paretoPoints[:, 1],
        mode='lines+markers',
        marker=dict(size=8, color='red'),
        line=dict(width=2, color='red'),
        
        name='Pareto Front'
    ))

    # Final bid
    agreedBid = all_utils[-1]
    fig.add_trace(go.Scatter(
        x=[agreedBid[0]], y=[agreedBid[1]],
        mode='markers',
        marker=dict(size=10, color='purple', symbol='diamond'),
        name='Final agreed upon bid'
    ))

    # Dotted line to closest Pareto point
    closestParetoPair = closestPoint(agreedBid, paretoPoints)
    distance = dist(agreedBid, closestParetoPair)
    fig.add_trace(go.Scatter(
        x=[agreedBid[0], closestParetoPair[0]],
        y=[agreedBid[1], closestParetoPair[1]],
        mode='lines',
        line=dict(dash='dot', color='grey'),
        name='Distance to Pareto: ' + str(distance),
        showlegend=True
    ))

    fig.update_layout(
        title='Negotiation Bids with Pareto Front and Final Offer',
        xaxis_title='Utility: ' + agent_names[1],
        yaxis_title='Utility: ' +agent_names[0],
        legend= dict( # allign legend to the right of the graph
            x=1.02,  
            y=1,
            xanchor='left',
            yanchor='top',
            bordercolor='black',
            borderwidth=1
        )
        #dict(x=0.8, y=0.99)
    )

    # Save or show
    if saveFig:
        os.makedirs(save_dir, exist_ok=True)
        name = os.path.basename(os.path.normpath(data_dir))
        #fig.write_image(os.path.join(save_dir, name + ".png"))
        fig.write_html(os.path.join(save_dir, name + ".html"))
    fig.show()

if __name__ == "__main__":
    defaultSaveDir = "figures/"
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Path to data directory")
    parser.add_argument("--saveFig", type=bool, default=True, help="Whether to save the figure")
    parser.add_argument("--save_dir", type=str, default=defaultSaveDir, help="Directory to save figure ")
    args = parser.parse_args()

    plot(args.dir, args.saveFig, args.save_dir)
