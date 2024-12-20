import argparse
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from GraphTsetlinMachine.graphs import Graphs
from GraphTsetlinMachine.tm import MultiClassGraphTsetlinMachine

def parse_arguments(**kwargs):
    parser = argparse.ArgumentParser(description="Train a Graph Tsetlin Machine.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--epochs", default=100, type=int, help="Number of training epochs.")
    parser.add_argument("--number-of-clauses", default=18, type=int, help="Number of clauses.")
    parser.add_argument("--T", default=400, type=int, help="Threshold T.")
    parser.add_argument("--s", default=0.8, type=float, help="Specificity parameter.")
    parser.add_argument("--depth", default=3, type=int, help="Graph depth.")
    parser.add_argument("--hypervector-size", default=512, type=int, help="Hypervector size.")
    parser.add_argument("--hypervector-bits", default=2, type=int, help="Number of bits per hypervector.")
    parser.add_argument("--message-size", default=512, type=int, help="Message size.")
    parser.add_argument("--message-bits", default=2, type=int, help="Number of bits per message.")
    parser.add_argument('--double-hashing', dest='double_hashing', action='store_true', help="Enable double hashing.")
    parser.add_argument("--max-included-literals", default=16, type=int, help="Maximum included literals.")
    
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

def load_data(file_path, train_split=0.9):
    data = pd.read_csv(file_path)
    subset_size = int(data.shape[0] * train_split)
    X = data.iloc[:subset_size, 0].values
    X_test = data.iloc[subset_size:, 0].values
    y = data.iloc[:subset_size, 1].values
    y_test = data.iloc[subset_size:, 1].values
    return X, X_test, y, y_test

def calculate_edges(board_size):
    edges = []
    for i in range(board_size):
        for j in range(board_size):
            # Add row edges
            if j < board_size - 1:
                edges.append(((i, j), (i, j + 1)))  # Right neighbor
            # Add column edges
            if i < board_size - 1:
                edges.append(((i, j), (i + 1, j)))  # Bottom neighbor
    return edges

def calculate_node_neighbors(board_size):
    n_edges_list = []
    for i in range(board_size**2):
        row, col = divmod(i, board_size)
        neighbors = 0
        if row > 0: neighbors += 1  # Top
        if row < board_size - 1: neighbors += 1  # Bottom
        if col > 0: neighbors += 1  # Left
        if col < board_size - 1: neighbors += 1  # Right
        n_edges_list.append(neighbors)
    return n_edges_list

def position_to_edge_id(pos, board_size):
    return pos[0] * board_size + pos[1]

def prepare_graphs(graphs, data, n_edges_list, edges, board_size):
    # Prepare nodes
    for graph_id in range(data.shape[0]):
        graphs.set_number_of_graph_nodes(
            graph_id=graph_id,
            number_of_graph_nodes=board_size**2,
        )
    graphs.prepare_node_configuration()

    # Add nodes
    for graph_id in range(data.shape[0]):
        for k in range(board_size**2):
            graphs.add_graph_node(graph_id, k, n_edges_list[k])

    # Prepare edges
    graphs.prepare_edge_configuration()

    # Add graph node properties and edges
    for graph_id in range(data.shape[0]):
        for k in range(board_size**2):
            graphs.add_graph_node_property(graph_id, k, data[graph_id][k])
        for edge in edges:
            node_id = position_to_edge_id(edge[0], board_size)
            destination_node_id = position_to_edge_id(edge[1], board_size)
            graphs.add_graph_node_edge(graph_id, node_id, destination_node_id, edge_type_name=0)
            graphs.add_graph_node_edge(graph_id, destination_node_id, node_id, edge_type_name=0)

    graphs.encode()    


def train_tsetlin_machine(args, graphs_train, y, graphs_test, y_test):
    tm = MultiClassGraphTsetlinMachine(
        args.number_of_clauses, args.T, args.s,
        depth=args.depth, message_size=args.message_size,
        message_bits=args.message_bits, max_included_literals=args.max_included_literals,
        grid=(16*13,1,1), block=(128,1,1)
    )
    epoch_progress = []
    start_time = time()
    for epoch in range(args.epochs):
        tm.fit(graphs_train, y, epochs=1, incremental=True)
        epoch_progress.append(np.mean(y_test == tm.predict(graphs_test)))
    duration = time() - start_time
    return tm, epoch_progress, duration

def main():
    args = parse_arguments()

    X, X_test, y, y_test = load_data('./3x3_small.csv')
    board_size = 3

    edges = calculate_edges(board_size)
    n_edges_list = calculate_node_neighbors(board_size)

    graphs_train = Graphs(
        number_of_graphs=len(X),
        symbols=["O", "X", " "],
        hypervector_size=args.hypervector_size,
        hypervector_bits=args.hypervector_bits,
        double_hashing=args.double_hashing,
    )
    prepare_graphs(graphs_train, X, n_edges_list, edges, board_size)

    graphs_test = Graphs(len(X_test), init_with=graphs_train)
    prepare_graphs(graphs_test, X_test, n_edges_list, edges, board_size)

    tm, epoch_progress, duration = train_tsetlin_machine(args, graphs_train, y, graphs_test, y_test)

    avg_accuracy = np.mean(epoch_progress)
    max_accuracy = np.max(epoch_progress)
    max_epoch = np.argmax(epoch_progress)

    print(*epoch_progress)
    print(f"{avg_accuracy} {max_accuracy} {max_epoch}")

if __name__ == "__main__":
    main()
