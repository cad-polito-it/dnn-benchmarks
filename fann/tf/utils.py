import os
import tensorflow as tf
import keras
import importlib.resources
from keras.src.layers import Layer
from typing import Callable, Optional


MODULE_PATH = importlib.resources.files(__package__)


def load_converted_tf_network(
    network_name: str, dataset_name: str, models_path: str = "../benchmark_models/models/pretrained_weights"
) -> keras.Model:
    """
    Load the converted keras network from .keras local file.

    The models are located in a directory with path ``models_path`` that is structured like this:
    - ``models_path``:
        - ``dataset_1``
            - ``model_1.keras``
            - ``model_2.keras``
        - dataset_2
            - ``model_3.keras``
            ...

    For example for retrieving model_3 in the default models folder (models/converted-tf) one should use the the following code:
    ```
        load_converted_tf_network("dataset_1", "model_3")
    ```

    ## Args
        - ``network_name``: The name of the network to load
        - ``dataset_name``: The name of the dataset used by the model

    """

    # Load the entire model with weights
    network_path = os.path.join(
        MODULE_PATH, models_path, dataset_name, f"{network_name}.keras"
    )

    return keras.models.load_model(network_path)


def deep_clone_function_factory(
    inner_clone_function: Callable[[Layer, Layer], Optional[Layer]],
    verbose=False,
    copy_weights=True,
) -> Callable[[Layer], Layer]:
    """
    Builds a Clone Function that can be used as the ```clone_function``` argument inside the ```keras.models.clone_model()```.
    The function built from this function is applied to all layer in the model, including also layers inside submodels, without limits
    on the depth of the model. The function allows to make changes on the new graph while cloning the old graph.

    ## Args
        - ```inner_clone_function```: A clone function that is slightly different from the one of ```keras.models.clone_model()```. It takes
            in input two ```keras.Layer``` objects, ```cloned_layer``` and ```old_layer```. The first one, ```cloned_layer``` is a layer cloned from the old model
            and depending on the returned object of this function may or may not be inserted in the cloned graph as it is.
            ```old_layer``` is the layer object from the old graph, and can be used to access properties
            of the graph to be cloned. The function returns a ```keras.Layer``` object, or ```None```. If it returns ```None``` or exactly ```cloned_layer``` then
            the layer being cloned is not changed. If it returns instead another layer object then the layer in the cloned model is changed withe layer returned.
            NOTE: Never return ```old_layer``` in this function.
        - ```verbose```: If ```True```, the function will log information about the current layer being cloned. Defaults to ```False```.
        - ```copy_weights```: If ```True``` the function copies automatically the weights from all the old layers to the new ones.

    ## Returns:
        A function that can be used as the ```clone_function``` argument in ```keras.models.clone_model()```, making sure that all layers all the models are
        inspected, including ones in the submodels, and depending on ```innner_clone_function``` changed to a new layer.

    ## Usage Example:
    ```
    import keras
    from tf_utils import deep_clone_function_factory

    class FaultInjector(keras.Layer):
        pass

    # Example of a function for adding a fault injection layer after a layer named "conv_2d_1"
    def inner_clone_function(cloned_layer, old_layer):
        if old_layer.name == "conv_2d_1":
            return keras.Sequentual([cloned_layer, FaultInjector()])
        else:
            return None #or cloned_layer

    model_to_clone = ...  # A keras Model
    clone_fn = deep_clone_function_factory(inner_clone_function, verbose=True)

    cloned_model = keras.models.clone_model(model_to_clone, clone_fn)
    ```

    """

    def _clone_function(layer):
        if verbose:
            print(f"Cloning Layer Name: {layer.name} Type:{type(layer)}")

        if isinstance(layer, keras.Model):
            if verbose:
                print(f"Layer {layer.name} is a sub-Model. Cloning it recursively")
            cloned_submodel = keras.models.clone_model(
                layer, clone_function=_clone_function
            )
            if copy_weights:
                cloned_submodel.set_weights(layer.get_weights())
            return cloned_submodel
        cloned_layer = layer.__class__.from_config(layer.get_config())
        maybe_changed_layer = inner_clone_function(cloned_layer, layer)
        if maybe_changed_layer is layer:
            raise ValueError(
                "Clone function returned the old layer. Never return the old_layer from the clone function."
            )
        if maybe_changed_layer is not None:
            return maybe_changed_layer
        else:
            return cloned_layer

    return _clone_function


def create_manipulated_model(
    model: keras.Model,
    clone_function: Callable[
        [Layer, Layer], Optional[Layer]
    ] = lambda cloned_layer, old_layer: None,
    verbose=False,
    copy_weights=True,
) -> keras.Model:
    """
    Clones and manipulate a Keras functional model, applying the clone_function also to nested submodels.

    This function is a wrapper for ```keras.models.clone_model()```

    ## Args:
        - ```model```: A Keras Functional model that will be cloned. This model will not be changed in the process.

        - ```clone_function```: An optional function that can make changes in the cloned model, with respect to the original function.
            If not specified the model will be cloned as it is without modifications.
            The function takes in input two keras.Layer (```cloned_layer, old_layer```) and returns a ```keras.Layer``` or ```None```.
            The first argument contains the cloned layer that would be placed in the new model unless another layer is returned in this function.
            The second argument contains the layer to be cloned from the old graph. That can be used to get information about the old model that is going
            to be cloned.
            If the function returns ```None``` or exactly ```cloned_layer``` then ```new_layer``` in the source model will be inserted as it is in the new cloned model
            If the function returns another layer, then that layer replaces the ```new_layer``` in the cloned model with a custom layer.
            NOTE: Never return ```old_layer``` from this function, otherwise a ```ValueError``` will be thrown.

        - ```verbose```: Print to stdout information about the layers when they are cloned. Defaults to ```False```.

        - ```copy_weights```: Copy the weights of the old model in the cloned model. This may throw errors if ```clone_function``` inserts layers with weights. Defaults to ```True```.


    ## Usage Example:
    ```
    import keras
    from tf_utils import create_manipulated_model

    class FaultInjector(keras.Layer):
        pass

    # Example of a function for adding a fault injection layer after a layer named "conv_2d_1"
    def clone_function(cloned_layer, old_layer):
        if old_layer.name == "conv_2d_1":
            return keras.Sequentual([cloned_layer, FaultInjector()])
        else:
            return None #or cloned_layer

    model_to_clone = ...

    cloned_model = create_manipulated_model(model_to_clone, clone_function)
    ```
    """
    clone_fn = deep_clone_function_factory(
        clone_function, verbose=verbose, copy_weights=copy_weights
    )
    cloned_model = keras.models.clone_model(model, clone_function=clone_fn)
    if copy_weights:
        cloned_model.set_weights(model.get_weights())
    return cloned_model
