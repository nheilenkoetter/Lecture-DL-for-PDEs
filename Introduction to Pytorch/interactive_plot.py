import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# def print_backward_graph(tensor, indent=0):
#     for f, _ in tensor.next_functions:
#         if f is not None:
#             print(f)
#             print_backward_graph(f, indent + 1)


def interactive_plot(test_data=False):
    # ReLU function
    def relu(x):
        return np.clip(x, 0.0, 1000.0)

    # Example data points
    x_data = np.linspace(-5, 5, 100)
    x_target = np.array([-4.0, -3, -2, -1, 0, 1, 2, 3, 4])
    y_target = np.array([2, 1.6, 1.0, 0.9, 1.1, 1.0, 2.3, 5.5, 6.8])

    x_test = np.array([2.5, 5])
    y_test = np.array([4.0, 6.7])

    # Forward pass for 2-layer network with 2 hidden neurons and independent output weights
    def relu_net_2layer(x, w11, b11, w12, b12, w2_1, w2_2, b2):
        h1_1 = relu(w11 * x + b11)
        h1_2 = relu(w12 * x + b12)
        y = w2_1 * h1_1 + w2_2 * h1_2 + b2
        return y

    # Plot function with MSE
    def plot_net(w1_1, b1_1, w1_2, b1_2, w2_1, w2_2, b2):
        y_plot = relu_net_2layer(x_data, w1_1, b1_1, w1_2, b1_2, w2_1, w2_2, b2)
        y_pred = relu_net_2layer(x_target, w1_1, b1_1, w1_2, b1_2, w2_1, w2_2, b2)
        mse = np.mean((y_pred - y_target)**2)
        
        plt.figure(figsize=(6,4))
        plt.plot(x_target, y_target, 'ro', label='Data')
        if test_data:
            plt.plot(x_test, y_test, 'go', label='Test Data')
        plt.plot(x_data, y_plot, 'b-', label='Network')
        plt.title(f"MSE: {mse:.4f}")
        plt.ylim(-1, 10)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()

    # Create sliders
    w11_slider = widgets.FloatSlider(min=-3, max=3, step=0.1, value=1, description='w_11')
    b11_slider = widgets.FloatSlider(min=-5, max=5, step=0.1, value=0, description='b_11')
    w12_slider = widgets.FloatSlider(min=-3, max=3, step=0.1, value=1, description='w_12')
    b12_slider = widgets.FloatSlider(min=-5, max=5, step=0.1, value=0, description='b_12')
    w2_1_slider = widgets.FloatSlider(min=-3, max=3, step=0.1, value=1, description='w_21')
    w2_2_slider = widgets.FloatSlider(min=-3, max=3, step=0.1, value=1, description='w_22')
    b2_slider = widgets.FloatSlider(min=-5, max=5, step=0.1, value=0, description='b_2')

    # Arrange sliders in two columns
    col1 = widgets.VBox([w11_slider, b11_slider, w12_slider, b12_slider])
    col2 = widgets.VBox([w2_1_slider, w2_2_slider, b2_slider])
    ui = widgets.HBox([col1, col2])

    # Connect sliders to plotting function
    out = widgets.interactive_output(plot_net, {
        'w1_1': w11_slider, 'b1_1': b11_slider,
        'w1_2': w12_slider, 'b1_2': b12_slider,
        'w2_1': w2_1_slider, 'w2_2': w2_2_slider,
        'b2': b2_slider
    })

    # Display the interactive UI
    display(ui, out)