import os
import sys
sys.path.append('.')
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from utils import calculate_second_node_coords, check_colliding_nodes


def read_gaussian_input(file_path):
    """
    Reads a Gaussian input file and extracts the molecular geometry.

    Parameters:
    file_path (str): Path to the Gaussian input file.

    Returns:
    list: A list of tuples containing the atomic symbols and their coordinates.
    """
    molecule = []
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Flag to indicate when to start reading the molecule
        read_molecule = False

        for line in lines:
            # Skip comments and empty lines
            if line.strip().startswith('#') or not line.strip():
                continue

            # The molecule section starts after the route section and the title section
            if not read_molecule:
                if line.strip() == '':
                    continue
                else:
                    # After the route and title, the next non-empty line is the charge and multiplicity
                    # The molecule section starts after that
                    read_molecule = True
                    continue

            # Split the line into components
            parts = line.split()
            if len(parts) < 4:
                continue

            # Extract atomic symbol and coordinates
            atom_symbol = parts[0]
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                molecule.append((atom_symbol, (x, y, z)))
            except ValueError:
                # Skip lines that don't contain valid coordinates
                continue

    return molecule


def create_molecular_graph(molecule):
    """
    Creates a molecular graph using networkx.

    Parameters:
    molecule (list): A list of tuples containing the atomic symbols and their coordinates.

    Returns:
    networkx.Graph: A graph representing the molecule.
    """
    G = nx.Graph()

    # Add nodes (atoms) to the graph
    for i, (atom, coords) in enumerate(molecule):
        G.add_node(i, symbol=atom, coords=coords)

    # Add edges (bonds) between atoms based on distance
    for i in range(len(molecule)):
        for j in range(i + 1, len(molecule)):
            # Calculate Euclidean distance between atoms
            x1, y1, z1 = molecule[i][1]
            x2, y2, z2 = molecule[j][1]
            distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5

            # Add an edge if the distance is within a typical bond length range
            if distance < 1.8:  # Adjust this threshold as needed
                G.add_edge(i, j)

    return G


def display_molecular_graph_3d(G):
    """
    Displays the molecular graph in 3D using matplotlib.

    Parameters:
    G (networkx.Graph): The molecular graph.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract node positions and labels
    pos = {i: G.nodes[i]['coords'] for i in G.nodes}
    labels = {i: G.nodes[i]['symbol'] for i in G.nodes}

    # Plot nodes
    for i in G.nodes:
        x, y, z = pos[i]
        ax.scatter(x, y, z, color='lightblue', s=1000, edgecolor='black')
        ax.text(x, y, z, labels[i], fontsize=12, ha='center', va='center')

    # Plot edges
    for i, j in G.edges:
        x1, y1, z1 = pos[i]
        x2, y2, z2 = pos[j]
        ax.plot([x1, x2], [y1, y2], [z1, z2], color='black')

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title("3D Molecular Graph")
    plt.show()


def create_functional_group_graph(functional_group, bond_lengths, bond_angles):
    """
    Creates a graph representing the functional group.

    Parameters:
    functional_group (list): A list of atomic symbols in the functional group.
    bond_lengths (dict): A dictionary of bond lengths between atoms.
    bond_angles (dict): A dictionary of bond angles between atoms.

    Returns:
    networkx.Graph: A graph representing the functional group.
    """
    G = nx.Graph()

    # Add nodes (atoms) to the graph
    for i, atom_symbol in enumerate(functional_group):
        G.add_node(i, symbol=atom_symbol, coords=(0.0, 0.0, 0.0))  # Placeholder coordinates

    # Add edges (bonds) between atoms
    for (i, j), length in bond_lengths.items():
        G.add_edge(i, j, length=length)

    # Calculate coordinates for the functional group
    for i in G.nodes:
        if i == 0:
            continue  # The first atom is at the origin
        for neighbor in G.neighbors(i):
            if neighbor < i:
                # Calculate coordinates based on bond length and angle
                direction_vector = np.random.rand(3) - 0.5  # Random initial direction
                direction_vector = direction_vector / np.linalg.norm(direction_vector)
                new_coords = calculate_second_node_coords(G, G.nodes[neighbor]['coords'], G.edges[i, neighbor]['length'], direction_vector)
                G.nodes[i]['coords'] = new_coords

    return G


def substitute_with_functional_group(G, index, functional_group_graph, attachment_atom, threshold=0.1):
    """
    Substitutes an atom at the specified index with a functional group, ensuring no collisions.

    Parameters:
    G (networkx.Graph): The molecular graph.
    index (int): The index of the atom to replace.
    functional_group_graph (networkx.Graph): The graph representing the functional group.
    attachment_atom (int): The index of the atom in the functional group to attach to the molecule.
    threshold (float): Minimum distance to avoid collisions (default: 0.1 Å).

    Returns:
    networkx.Graph: The modified graph with the functional group added.
    """
    # Get the coordinates of the atom being replaced
    replaced_atom_coords = G.nodes[index]['coords']

    # Replace the atom with the attachment atom of the functional group
    G.nodes[index]['symbol'] = functional_group_graph.nodes[attachment_atom]['symbol']
    G.nodes[index]['coords'] = replaced_atom_coords

    # Add the remaining atoms of the functional group
    new_node_indices = []
    for i in functional_group_graph.nodes:
        if i == attachment_atom:
            continue  # Skip the attachment atom (already replaced)

        # Calculate coordinates for the new node relative to the replaced atom's coordinates
        bond_length = functional_group_graph.edges[attachment_atom, i]['length']
        direction_vector = functional_group_graph.nodes[i]['coords']  # Relative coordinates from the functional group graph
        new_coords = (
            replaced_atom_coords[0] + direction_vector[0] * bond_length,
            replaced_atom_coords[1] + direction_vector[1] * bond_length,
            replaced_atom_coords[2] + direction_vector[2] * bond_length
        )

        # Check for collisions
        if check_colliding_nodes(G, threshold):
            raise ValueError("Collision detected. Could not find a valid position for the new node.")

        # Add the new node
        new_index = max(G.nodes) + 1 if G.nodes else 0  # Assign a new unique index
        G.add_node(new_index, symbol=functional_group_graph.nodes[i]['symbol'], coords=new_coords)
        new_node_indices.append(new_index)

    # Connect the functional group to the rest of the molecule
    for neighbor in G.nodes:
        if neighbor != index:  # Avoid connecting to the replaced node
            G.add_edge(index, neighbor)  # Connect the attachment atom to the rest of the molecule

    # Add edges within the functional group
    for i, j in functional_group_graph.edges:
        if i == attachment_atom:
            G.add_edge(index, new_node_indices[j - 1])
        elif j == attachment_atom:
            G.add_edge(index, new_node_indices[i - 1])
        else:
            G.add_edge(new_node_indices[i - 1], new_node_indices[j - 1])

    return G


if __name__ == "__main__":
    # Define the input file path
    input_file = "Substitutor/Example_Data/CH4.gjf"  # Replace with your Gaussian input file path

    # Read the molecule from the input file
    molecule = read_gaussian_input(input_file)

    # Create the molecular graph
    G = create_molecular_graph(molecule)

    # Display the molecular graph in 3D
    display_molecular_graph_3d(G)

    # Ask the user to specify which atom to replace and define the functional group
    print("Molecular graph displayed. Please inspect the graph and specify the index of the atom to replace.")
    index_to_replace = int(input("Enter the index of the atom to replace: "))

    # Define the functional group (e.g., CO2)
    functional_group = ['C', 'O', 'O']
    bond_lengths = {(0, 1): 1.16, (0, 2): 1.16}  # C=O bond lengths
    bond_angles = {(0, 1, 2): 180.0}  # Linear geometry for CO2

    # Create the functional group graph
    functional_group_graph = create_functional_group_graph(functional_group, bond_lengths, bond_angles)

    # Specify the attachment atom in the functional group (e.g., C in CO2)
    attachment_atom = 0  # Index of the attachment atom in the functional group

    # Substitute the specified atom with the functional group
    try:
        G = substitute_with_functional_group(G, index_to_replace, functional_group_graph, attachment_atom)
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Display the modified molecular graph in 3D
    display_molecular_graph_3d(G)

    # Write the modified molecule to a new Gaussian input file
    output_file = 'modified_gaussian_file.gjf'  # Output file with the functional group added
    route_section = "%chk=example.chk\n%mem=2GB\n%nprocshared=4\n#p B3LYP/6-31G(d) Opt"
    title_section = "Title Card Required"
    charge = 0  # Replace with the correct charge
    multiplicity = 1  # Replace with the correct multiplicity

    with open(output_file, 'w') as file:
        # Write the route section
        file.write(route_section + '\n\n')

        # Write the title section
        file.write(title_section + '\n\n')

        # Write the charge and multiplicity
        file.write(f"{charge} {multiplicity}\n")

        # Write the molecule
        for i in G.nodes:
            atom_symbol = G.nodes[i]['symbol']
            x, y, z = G.nodes[i]['coords']
            file.write(f"{atom_symbol:2s} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    print(f"Modified molecule written to {output_file}")
