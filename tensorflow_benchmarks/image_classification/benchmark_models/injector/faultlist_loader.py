import csv
from typing import List, Sequence, Tuple


def convert_weights_coords_from_pt_to_tf(weight_coords):
    """
    Convert the order of the weights from the PyTorch axis order to TensorFlow axis order.

    Args
    ---
    * ``weight_coords : Iterable[int]``
    A tuple (or list) of integer indicating the coordinates to read from a weight in PyTorch format.
    For now only 2d and 4d coordinates are supported.

    Return
    ---
    A tuple of 2 or 4 ints (same as the input), representing the converted weights coordinate to PyTorch.

    For a 2D tensor used in a Linear layer, the function swaps the weight coordinates
    from PyTorch format ``(out, in)`` to Keras format ``(in, out)``
    where:
     * ``in`` represents the input feature to which the weight is applied
     * ``out`` is the output feature targeted

    For 4D tensors used in Convolutions the function swaps the weight coordinates from the
    PyTorch convention ``(out, in, filter_x, filter_y)`` to Keras convention ``(filter_x, filter_y, in, out)``
    where:
    * ``in`` ``out`` are similar to the 2D case
    * ``filter_x`` ``filter_y`` are the coordinate inside each filter
    """
    if len(weight_coords) == 2:
        out_features, in_features = weight_coords
        return (in_features, out_features)
    if len(weight_coords) == 4:
        out_channels, in_channels, filter_height, filter_width = weight_coords
        return (filter_height, filter_width, in_channels, out_channels)
    raise NotImplementedError(
        f"The rearrangement when weights have {len(weight_coords)} dimensions is not handled"
    )


def load_fault_list(fault_list_path: str, convert_faults_pt_to_tf=False) -> Tuple[List[str], List[Tuple[int, str, Sequence, int]]]:
    """
    Loads fault list from a .csv file. The fault list is assumed to have coordinates in
    PyTorch coordinates.

    A faultlist is a CSV file, with a comma separator, file with the following headings:
     * ``Injection``: The progressive, zero-indexed id of the injection
     * ``Layer``: The fully qualified name of the PyTorch layer, so that from the name
                  the layer can be accessed using the ``nn.Module.get_submodule`` method
    * ``TensorIndex``: The index of the weight of the layer, using PyTorch axis order. Note that biases are not considered.
    * ``Bit``: An integer indicating the bit position to flip. For float32, 0 is the LSB of signifcand, 30 is the MSB of exponent, 31 the sign

    The headings are specified in the first row of the file.

    Faultlist Example
    ----
    ```csv
    Injection,Layer,TensorIndex,Bit
    0,conv1.conv,"(62, 2, 0, 2)",30
    1,inception3a.branch1.conv,"(9, 155, 0, 0)",10
    ```


    Args
    ---
    * ``fault_list_path: str``
    A path string pointing to the .csv faultlist file to be loaded

    * ``convert_faults_pt_to_tf : bool`` (default is ``False``)
    If ``True`` the PyTorch weights coordinate  in each injection of the fault list will be rearranged to be
    according to the Keras axis ordering

    Returns
    ---
    A Tuple of two items
    1. The list of uniques names of all layers encountered in the faultlist in the order in which they are encountered.
    2. A List of Tuples. Each item of the list contains details of one injection.
    Each tuple contains ``(Injection, Layer, TensorIndex, Bit)``, converted to their types.

    """
    layer_list = []
    injections = []
    with open(fault_list_path) as f:
        cr = csv.reader(f)
        next(cr, None)  # Skip Header
        for row in cr:
            try:
                inj_id, layer_name, weight_pos, bit = row
                inj_id = int(inj_id)
                if layer_name not in layer_list:
                    layer_list.append(layer_name)

                weight_pos = [
                    int(coord.strip()) for coord in weight_pos.strip("()").split(",")
                ]
                if convert_faults_pt_to_tf:
                    weight_pos = convert_weights_coords_from_pt_to_tf(weight_pos)
                bit = int(bit)
                injections.append((inj_id, layer_name, weight_pos, bit))
            except Exception as e:
                print(f"Error happened processing the following row: {row}")
                raise e from None
        return layer_list, injections
