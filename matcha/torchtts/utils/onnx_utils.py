import logging

from onnx import helper as xhelp
from onnx import onnx_ml_pb2 as xpb2

NODE_COUNT = 1
logger = logging.getLogger(__name__)

# Data types, see https://deeplearning4j.org/api/latest/onnx/Onnx.TensorProto.DataType.html
DATA_TYPES = {
    "FLOAT": 1,
    "UINT8": 2,
    "INT8": 3,
    "UINT16": 4,
    "INT16": 5,
    "INT32": 6,
    "INT64": 7,
    # "STRING" : 8,
    "BOOL": 9,
    "FLOAT16": 10,
    "DOUBLE": 11,
    "UINT32": 12,
    "UINT64": 13,
    "COMPLEX64": 14,
    "COMPLEX128": 15,
}


def _data_type(data_string: str):
    """convert the data type string (i.e., FLOAT, INT16, etc.) to the appropriate int.
    See: https://deeplearning4j.org/api/latest/onnx/Onnx.TensorProto.DataType.html
    """
    for key, val in DATA_TYPES.items():
        if key == data_string:
            return val
    logger.error("Data string not found. Use `list_data_types()` to list all supported data strings.")
    return False


def create_graph_member_map(graph_member_list):
    member_map = {}
    for n in graph_member_list:
        member_map[n.name] = n
    return member_map


def add_input(graph: xpb2.GraphProto, name: str, data_type: str, dimensions: [], **kwargs):
    """Add an input to a graph
    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        name: String, the name of the input as used to determine the graph topology.
        data_type: String, the data type of the input. Run list_data_types() for an overview.
        dimensions: List[] specifying the dimensions of the input.
        **kwargs
    Returns:
        The extended graph.
    """
    if type(graph) is not xpb2.GraphProto:
        logger.error("graph is not a valid ONNX graph.")
        return False

    dtype = _data_type(data_type)
    if not dtype:
        return False

    try:
        graph.input.append(xhelp.make_tensor_value_info(name, dtype, dimensions, **kwargs), *kwargs)
    except Exception as e:
        logger.error("Unable to add the input: " + str(e))
        return False
    return graph


def make_node(op_type: str, inputs: [], outputs: [], name: str = "", **kwargs):
    """Create a new node
    Args:
        op_type: Operator type, see https://github.com/onnx/onnx/blob/master/docs/Operators.md
        inputs: [] list of inputs (names to determine the graph topology)
        outputs: [] list of outputs (names to determine the graph topology)
        name: The name of this node (Optional)
        **kwargs
    """
    if not name:
        global NODE_COUNT
        name = "onnx-node" + str(NODE_COUNT)
        NODE_COUNT += 1

    try:
        node = xhelp.make_node(op_type, inputs, outputs, name, **kwargs)
    except Exception as e:
        logger.error("Unable to create node: " + str(e))
        return False
    return node


def add_node(graph: xpb2.GraphProto, node: xpb2.NodeProto, **kwargs):
    """Add node appends a node to graph g and returns the extended graph
    Prints a message and returns False if fails.
    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        node: A node, onnx.onnx_ml_pb2.NodeProto.
        **kwargs
    Returns:
        The extended graph.
    """
    if type(graph) is not xpb2.GraphProto:
        logger.error("The graph is not a valid ONNX graph.")
        return False

    if type(node) is not xpb2.NodeProto:
        logger.error("The node is not a valid ONNX node.")
        return False

    try:
        graph.node.append(node, **kwargs)
    except Exception as e:
        logger.error("Unable to extend graph: " + str(e))
        return False
    return graph


def delete_node(graph: xpb2.GraphProto, node_name: str):
    """Delete a node to graph g and returns the graph
    Prints a message and returns False if fails.
    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        node_name: Name of the node to remove.
    Returns:
        The extended graph.
    """
    if type(graph) is not xpb2.GraphProto:
        logger.error("The graph is not a valid ONNX graph.")
        return False

    if not node_name:
        logger.error("Please specify a node name.")
        return False

    found = False
    try:
        for elem in graph.node:
            if elem.name == node_name:
                graph.node.remove(elem)
                found = True
    except Exception as e:
        logger.error("Unable to iterate the nodes. " + str(e))
        return False
    if not found:
        logger.error("Unable to find the node by name.")
        return False

    return graph


def replace_node(graph: xpb2.GraphProto, node_name: str, node: xpb2.NodeProto):
    """Replace the node of graph g with given name to a new node
    Prints a message and returns False if fails.
    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        node_name: Name of the node to remove.
        node: A node, onnx.onnx_ml_pb2.NodeProto.
    Returns:
        The extended graph.
    """
    # We need to insert the new node to previous node index to make sure
    # the node list graph.node is still topological sorted after replacing.
    index = -1
    for i, elem in enumerate(graph.node):
        if elem.name == node_name:
            graph.node.remove(elem)
            index = i
    if index == -1:
        logger.error("Unable to find the node by name.")
        return False

    try:
        graph.node.insert(index, node)
    except Exception as e:
        logger.error("Unable to replace node: " + str(e))
        return False

    return graph


def delete_initializer(graph: xpb2.GraphProto, initializer_name: str):
    """Delete an initializer to graph g and returns the graph
    Prints a message and returns False if fails.
    Args:
        graph: A graph, onnx.onnx_ml_pb2.GraphProto.
        initializer_name: Name of the node to remove.
    Returns:
        The extended graph.
    """
    if type(graph) is not xpb2.GraphProto:
        logger.error("The graph is not a valid ONNX graph.")
        return False

    if not initializer_name:
        logger.error("Please specify a initializer name.")
        return False

    found = False
    try:
        for elem in graph.initializer:
            # print(elem.name)
            if elem.name == initializer_name:
                graph.initializer.remove(elem)
                found = True
    except Exception as e:
        logger.error("Unable to iterate the nodes. " + str(e))
        return False
    if not found:
        logger.error("Unable to find the node by name.")
        return False

    return graph
