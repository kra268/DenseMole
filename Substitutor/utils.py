import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


def display_molecular_graph(G):
    """
    Displays the molecular graph using matplotlib.

    Parameters:
    G (networkx.Graph): The molecular graph.
    """
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    plt.show()


def euclidean_distance(coords:list):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    coords (list): A list of coordinates (x, y, z) for two points.

    Returns:
    np.float: The Euclidean distance between the two points.
    """
    x1, y1, z1 = coords[0]
    x2, y2, z2 = coords[1]
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def calculate_neighbor_distances(G, node):
    """
    Calculates the distances between a node and its neighbors.

    Parameters:
    G (networkx.Graph): The graph.
    node (int): The node whose neighbor distances are to be calculated.

    Returns:
    dict: A dictionary where keys are neighbor nodes and values are distances.
    """
    distances = {}
    x1, y1, z1 = G.nodes[node]['coords']  # Coordinates of the central node

    for neighbor in G.neighbors(node):
        x2, y2, z2 = G.nodes[neighbor]['coords']  # Coordinates of the neighbor
        distance = euclidean_distance([(x1, y1, z1), (x2, y2, z2)])
        distances[neighbor] = distance

    return distances


def calculate_second_node_coords(G, first_node_coords, distance, direction_vector, threshold=0.1, max_attempts=100, attempt=0):
    """
    Calculates the coordinates of the second node, ensuring no collisions with existing nodes.

    Parameters:
    G (networkx.Graph): The graph.
    first_node_coords (tuple): Coordinates of the first node (x1, y1, z1).
    distance (float): Euclidean distance between the two nodes.
    direction_vector (tuple): Initial direction vector (dx, dy, dz).
    threshold (float): Minimum distance to avoid collisions (default: 0.1 Å).
    max_attempts (int): Maximum number of attempts to find a valid position (default: 100).
    attempt (int): Current attempt number (used for recursion).

    Returns:
    tuple: Coordinates of the second node (x2, y2, z2) or None if no valid position is found.
    """
    # Normalize the direction vector to ensure it's a unit vector
    direction_vector = np.array(direction_vector)
    direction_vector = direction_vector / np.linalg.norm(direction_vector)

    # Calculate the coordinates of the second node
    second_node_coords = tuple(np.array(first_node_coords) + distance * direction_vector)

    # Check for collisions
    colliding_pairs = check_colliding_nodes(G, threshold)

    # If no collisions, return the coordinates
    if not colliding_pairs:
        return second_node_coords

    # If collisions exist, generate a new direction vector and try again
    if attempt < max_attempts:
        print(f"Collision detected. Attempt {attempt + 1}/{max_attempts}: Adjusting direction vector.")
        # Generate a random direction vector
        new_direction_vector = np.random.rand(3) - 0.5  # Random vector in [-0.5, 0.5] range
        return calculate_second_node_coords(G, first_node_coords, distance, new_direction_vector, threshold, max_attempts, attempt + 1)
    else:
        print("Max attempts reached. Could not find a valid position.")
        return None


def check_colliding_nodes(G, threshold=0.5):
    """
    Checks if there are any colliding nodes in the graph.

    Parameters:
    G (networkx.Graph): The graph.
    threshold (float): The minimum distance between nodes to avoid collision (default: 0.5 Å).

    Returns:
    list: A list of tuples containing pairs of colliding nodes and their distances.
    """
    colliding_pairs = []

    # Get all node coordinates
    nodes = list(G.nodes(data=True))

    # Compare each pair of nodes
    for i in range(len(nodes)):
        node1, data1 = nodes[i]
        if 'coords' not in data1 or data1['coords'] is None:
            raise ValueError(f"Node {node1} is missing the 'coords' attribute or it is None.")

        x1, y1, z1 = data1['coords']

        for j in range(i + 1, len(nodes)):
            node2, data2 = nodes[j]
            if 'coords' not in data2 or data2['coords'] is None:
                raise ValueError(f"Node {node2} is missing the 'coords' attribute or it is None.")

            x2, y2, z2 = data2['coords']

            # Calculate Euclidean distance
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

            # Check if the distance is below the threshold
            if distance < threshold:
                colliding_pairs.append(((node1, node2), distance))

    return colliding_pairs