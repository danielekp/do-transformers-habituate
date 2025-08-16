from .visualization import plot_habituation_analysis
from .MLP_activation_hook import MLPActivationRecorder

def get_statistics(recorder: MLPActivationRecorder, prompt: str, target_token: str, layer_to_analyze: int):
    # Run the repeated prompt
    results = recorder.run_inference_with_hooks(prompt, target_token)
    assert layer_to_analyze < results["num_layers_hooked"]

    # Get a list of magnitudes for the interesting layers (best layer for target token)

    list_of_activations = []
    interesting_layers = [f"layer_{layer_to_analyze}_pos_{token_pos}" for token_pos in results['target_positions']]
    for layer in interesting_layers:
        list_of_activations.append(results["activations"][layer][0])

    # Plot and statistics

    statistics = plot_habituation_analysis(list_of_activations, target_token)

    return statistics