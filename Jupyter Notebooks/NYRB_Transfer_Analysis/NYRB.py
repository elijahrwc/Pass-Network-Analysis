"""
A Python Function Library for Mapping Passing Networks in Soccer Matches. 
Created by Elijah Weston-Capulong on February 24, 2025, for the NY Red Bulls Data Science team.
Modified: March 1, 2025

References:
- StatsBombPy: Their open-source Python library for accessing StatsBomb data.
- mplsoccer: A Python library for creating soccer visualizations.
- NetworkX: A Python library for creating and analyzing complex networks.
- Pandas, Numpy, Matplotlib: Common Python libraries for data manipulation and visualization.

This analysis explores how recent European signing, Eric Maxim Chupo-Moting, could impact the New York Red Bulls' tactical outlay,
setup, and performance through pass-network analysis of his event data available through the StatsBomb API.
"""

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

import networkx as nx
from statsbombpy import sb
from mplsoccer import Pitch, Sbopen

# Global variable for Chupo-Moting's player ID
MOTING_ID = 3499

def load_competitions():
    """
    Load competitions from StatsBomb data and filter for desired competitions.

    Args: None

    Returns:
        merged_data (DataFrame): DataFrame with data for selected competitions.
    """
    sb_data = sb.competitions()
    
    selected_competitions = [
        '1. Bundesliga',
        'Ligue 1',
        'African Cup of Nations',
        'FIFA World Cup',
        'Champions League' # These are the competitions in which Choup-Moting 
                           # has played that StatsBomb has data for.
    ]

    comp_frames = [sb_data.loc[sb_data['competition_name'] == comp] for comp
                    in selected_competitions]
    merged_data = pd.concat(comp_frames).reset_index(drop=True)

    return merged_data


def load_events_for_matches(match_ids, team_filter=None):
    """
    Load events for a list of match IDs.
    
    Parameters:
        match_ids (list): List of match IDs.
        team_filter (str, optional): Filter events by team name if provided.
    
    Returns:
        DataFrame: Concatenated events for the given match IDs.
    """
    parser = Sbopen()
    all_event_data = []
    for match_id in match_ids:
        events, related, freeze, tactics = parser.event(match_id)
        if team_filter:
            events = events.loc[events['team_name'] == team_filter]
        all_event_data.append(events)
    events_df = pd.concat(all_event_data, ignore_index=True)
    return events_df.reset_index(drop=True)


def filter_passes(df, success_only=True):
    """
    Filter an events dataframe for solely pass events.
    
    Parameters:
        df (DataFrame): The events dataframe to be filtered.
        success_only (bool): If True, filter for successful passes 
                            (i.e. where outcome_name is null).
    
    Returns:
        DataFrame: Filtered passes with selected columns.
    """
    passes = df[df['type_name'] == 'Pass']
    if success_only:
        passes = passes[passes['outcome_name'].isnull()]

    # Select relevant columns, these can be selected custom as needed
    return passes[['player_name', 'x', 'y', 'end_x', 'end_y', 
                   'pass_recipient_name', 'outcome_id', 'outcome_name', 
                   'player_id', 'pass_recipient_id', 'minute']]


def create_pass_network(pass_df):
    """
    Create a directed graph of passes from the given DataFrame.
    
    Parameters:
        pass_df (DataFrame): DataFrame with pass events.
    
    Returns:
        nx.DiGraph: Networkx directed graph of passes.
    """
    G = nx.DiGraph()
    for _, row in pass_df.iterrows():
        if (pd.notna(row['player_name']) 
            and pd.notna(row['pass_recipient_name'])):
            # Add nodes and update edge weight if edge already exists
            G.add_node(row['player_name'])
            G.add_node(row['pass_recipient_name'])
            if G.has_edge(row['player_name'], row['pass_recipient_name']):
                G[row['player_name']][row['pass_recipient_name']]['weight'] += 1.5
                # Customize added weight as needed
            else:
                G.add_edge(row['player_name'], row['pass_recipient_name'], 
                           weight=1) 
                # Lower weight for first pass adjusts for the likelihood the 
                # pass was in an unordinary sequence of events
    return G

def calculate_degrees(G):
    """
    Calculate and print in-degree, out-degree, and degree centrality for 
    graph G.

    Args: 
        G (nx.DiGraph): Directed graph of passes.
    
    Returns: None
    """
    in_degree = pd.Series(dict(G.in_degree())).sort_values(ascending=False)
    out_degree = pd.Series(dict(G.out_degree())).sort_values(ascending=False)
    degree_centrality = pd.Series(nx.degree_centrality(G)).sort_values(
                        ascending=False)
    
    print("In-Degree:")
    for i, (player, deg) in enumerate(in_degree.items(), start=1):
        print(f"{i}. Player: {player}: {deg}")
    
    print("\nOut-Degree:")
    for i, (player, deg) in enumerate(out_degree.items(), start=1):
        print(f"{i}. Player: {player}: {deg}")
    
    print("\nDegree Centrality:")
    for i, (player, cent) in enumerate(degree_centrality.items(), start=1):
        print(f"{i}. Player: {player}: {cent}")

def plot_pass_network(average_locations, pass_between, title, 
                      scaling=(1.2, 0.8), count_threshold=2):
    """
    Plot a pass network on a pitch using mplsoccer.
    
    Parameters:
        average_locations (DataFrame): DataFrame with each player's average x
                                      and y positions and pass counts.
        pass_between (DataFrame): DataFrame with pass counts between players.
        title (str): Plot title.
        scaling (tuple): Scaling factors for x and y coordinates.
        threshold (int): Minimum pass_count required to draw a line.
    """
    pitch, ax = nx.draw_pitch(pitch_type='statsbomb', pitch_color='grass', 
                              line_color='white', player_highlight_id=None)
    if count_threshold is not None:
        filtered = pass_between[pass_between['pass_count'] > count_threshold] 
        # Filter pass data based on threshold
    
    # Draw lines for passes
    ax = pitch.lines(
        scaling[0] * filtered.x, scaling[1] * filtered.y,
        scaling[0] * filtered.x_end, scaling[1] * filtered.y_end,
        lw=filtered.pass_count * 1.1, color='blue', zorder=1, ax=ax
    )
    
    # Draw nodes for players
    pitch.scatter(
        scaling[0] * average_locations.x, scaling[1] * average_locations.y,
        s=20 * average_locations['count'], color='white', edgecolors='#a6aab3',
        linewidth=2, alpha=1, zorder=1, ax=ax
    )
    
    # Annotate nodes with player names
    for player, row in average_locations.iterrows():
        pitch.annotate(
            player, xy=(scaling[0] * row.x, scaling[1] * row.y),
            c='#161A30', fontweight='bold', va='center', ha='center', size=8, 
            ax=ax
        )
    
    ax.set_title(title, color='black', fontsize=12, fontweight='bold', pad=20)
    plt.show()

def plot_pass_network_highlight(G, highlight_player, title='Pass Network', 
                                node_size=500, base_node_color='skyblue',
                                highlight_color='green', label_color='orange',
                                highlight_label_color='red'):
    """
    Plot a pass network graph with a specific player highlighted.
    
    Parameters:
        G (networkx.DiGraph): The directed pass network graph.
        highlight_player (str): The name of the player to highlight.
        title (str): The title for the plot.
        node_size (int): Size of the nodes.
        base_node_color (str): Color for non-highlighted nodes.
        highlight_color (str): Color for the highlighted node.
        label_color (str): Font color for labels of non-highlighted nodes.
        highlight_label_color (str): Font color for the highlighted player's
        label.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    plt.figure(figsize=(12, 8))
    
    # Compute layout and edge weights
    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    
    # Set node colors: highlighted if node matches, otherwise base color
    node_colors = [highlight_color if node == highlight_player 
                   else base_node_color for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors)
    
    # Draw edges with proportional width
    nx.draw_networkx_edges(G, pos, width=[w * 0.5 for w in weights], 
                           edge_color='gray')
    
    # Create label dictionary: separate for highlighted and non-highlighted
    # players
    labels = {node: node for node in G.nodes()}
    labels_non_highlight = {node: label for node, label in labels.items() 
                            if node != highlight_player}
    labels_highlight = ({highlight_player: highlight_player}
                         if highlight_player in G.nodes() else {})
    
    # Draw non-highlighted labels (small font, label_color)
    nx.draw_networkx_labels(G, pos, labels=labels_non_highlight, font_size=8,
                            font_color=label_color, font_weight='bold')
    
    # Draw highlighted label (larger font, highlight_label_color)
    if labels_highlight:
        nx.draw_networkx_labels(G, pos, labels=labels_highlight, font_size=10,
                                font_color=highlight_label_color,
                                font_weight='bold')
    
    plt.title(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

def get_jersey_data(match_id, parser=sb):
    """
    Fetches the lineup for the given match and returns jersey data.

    Args: 
        match_id (int): The match ID.
        parser (StatsBombParser): The parser object to use for data retrieval.
    
    Returns:
        DataFrame: DataFrame with player ID, jersey number, and player name.
    
    """
    lineups = parser.lineups(match_id)
    lineup_df = pd.concat(lineups, ignore_index=True)
    return lineup_df[['player_id', 'jersey_number', 'player_name']]

def create_pass_network_datasets(passes_df, events_df, match_id,
                                  subs_filter=True):
    """
    Creates pass network datasets from the given events DataFrame.

    Args:
        passes_df (DataFrame): DataFrame with pass events.
        events_df (DataFrame): DataFrame with all events.
        match_id (int): The match ID.
        subs_filter (bool): If True, filter events before the first 
        substitution.
    
    Returns:
        passes_df (DataFrame): DataFrame with pass events and player jersey 
        data.
        average_locations (DataFrame): DataFrame with average player locations.
    """

    # Get jersey data and merge for passers
    jersey_data = get_jersey_data(match_id)
    passes_df = pd.merge(passes_df, jersey_data, on='player_id', how='left')
    
    # Create a 'passer' column using the jersey_number from the merged data
    passes_df['passer'] = passes_df['jersey_number']
    
    # Use the actual pass recipient info available in the events DataFrame.
    # Create a mapping from player_name to player_id using jersey_data.
    name_to_id = jersey_data.set_index('player_name')['player_id'].to_dict()
    # Map pass_recipient_name to a new numeric column pass_recipient_id.
    passes_df['pass_recipient_id'] = (passes_df['pass_recipient_name'].
                                      map(name_to_id))
    
    # Merge to attach recipient jersey info.
    passes_df = pd.merge(
        passes_df,
        jersey_data[['player_id', 'jersey_number']],
        left_on='pass_recipient_id',
        right_on='player_id',
        how='left',
        suffixes=('', '_recipient')
    )
    
    # Rename the merged jersey number for recipients for clarity.
    passes_df.rename(columns={'jersey_number_recipient': 'pass_recipient'}, 
                     inplace=True)
    
    if subs_filter:
        # Filter for events before the first substitution (if any)
        subs = events_df[events_df['type_name'] == 'Substitution']['minute']
        if not subs.empty:
            first_sub = subs.min()
            passes_df = passes_df[passes_df['minute'] < first_sub]
    
    # Group by passer and calculate average locations
    average_locations = passes_df.groupby('passer').agg({'x': ['mean'], 'y': 
                                                         ['mean', 'count']})
    average_locations.columns = ['x', 'y', 'count']
    
    return passes_df, average_locations


def top_pass_betweens_for_player(player_id, pass_between_df, top_n=5):
    """
    Given a player_id and a DataFrame (with columns 'passer', 
    'pass_recipient', and 'pass_count'), return the top 'top_n' pass connections
    for that player(as a passer) sorted by the number of passes.
    
    Parameters:
    - player_id: Identifier for the player (should match values in the 'passer' 
    column)
    - pass_between_df: DataFrame containing the columns 'passer', 
    'pass_recipient', and 'pass_count'
    - top_n: Number of top connections to return (default: 5)
    
    Returns:
    - DataFrame of the top pass connections for the given player.
    """
    player_passes = pass_between_df[pass_between_df['passer'] == player_id]
    player_passes_sorted = player_passes.sort_values(by='pass_count', 
                                                     ascending=False)
    return player_passes_sorted.head(top_n)

