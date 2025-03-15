import os, sys
sys.path.append('.')
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


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


def calculate_relative_positions(central_atom_coords, bond_length, bond_angle):
    """
    Calculates the relative positions of atoms in a functional group.

    Parameters:
    central_atom_coords (tuple): Coordinates of the central atom.
    bond_length (float): Bond length between the central atom and the new atoms.
    bond_angle (float): Bond angle in degrees.

    Returns:
    list: A list of tuples containing the relative coordinates of the new atoms.
    """
    x0, y0, z0 = central_atom_coords
    bond_angle_rad = np.radians(bond_angle)

    # Calculate positions for a linear functional group (e.g., CO2)
    dx = bond_length * np.cos(bond_angle_rad)
    dy = bond_length * np.sin(bond_angle_rad)

    # Positions for two atoms in a linear arrangement
    pos1 = (x0 + dx, y0 + dy, z0)
    pos2 = (x0 - dx, y0 - dy, z0)

    return [pos1, pos2]


def substitute_with_functional_group(G, index, functional_group, bond_length, bond_angle):
    """
    Substitutes an atom at the specified index with a functional group.

    Parameters:
    G (networkx.Graph): The molecular graph.
    index (int): The index of the atom to replace.
    functional_group (list): A list of atomic symbols in the functional group.
    bond_length (float): Bond length between the central atom and the new atoms.
    bond_angle (float): Bond angle in degrees.

    Returns:
    networkx.Graph: The modified graph with the functional group added.
    """
    # Get the coordinates of the atom to replace
    central_atom_coords = G.nodes[index]['coords']

    # Calculate the relative positions of the new atoms
    relative_positions = calculate_relative_positions(central_atom_coords, bond_length, bond_angle)

    # Remove the atom at the specified index
    G.remove_node(index)

    # Add the functional group to the graph
    new_node_indices = []
    for i, atom_symbol in enumerate(functional_group):
        new_index = max(G.nodes) + 1 if G.nodes else 0  # Assign a new unique index
        G.add_node(new_index, symbol=atom_symbol, coords=relative_positions[i])
        new_node_indices.append(new_index)

    # Connect the functional group to the rest of the molecule
    for neighbor in G.nodes:
        if neighbor != index:  # Avoid connecting to the removed node
            G.add_edge(new_node_indices[0], neighbor)  # Connect the first atom of the functional group

    # Add edges within the functional group
    for i in range(1, len(new_node_indices)):
        G.add_edge(new_node_indices[0], new_node_indices[i])

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

    # Define the functional group
    functional_group = input("Enter the functional group (e.g., 'CO2'): ").strip().upper()

    # Define bond length and angle for the functional group
    if functional_group == 'CO2':
        bond_length = 1.16  # C=O bond length in Angstroms
        bond_angle = 180.0  # Linear geometry for CO2
    else:
        # Default values for other functional groups
        bond_length = 1.0  # Adjust as needed
        bond_angle = 120.0  # Adjust as needed

    # Substitute the specified atom with the functional group
    G = substitute_with_functional_group(G, index_to_replace, list(functional_group), bond_length, bond_angle)

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