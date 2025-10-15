from manim import *
import numpy as np
import pysindy as ps
data = np.load("data/double_pendulum_dataset.npz")
t = data['t']
Y_noisy = data["Y_noisy"]
Y_true = data["Y_true"]
sigmas = data["sigma"]
from sklearn.model_selection import train_test_split

train_size = 80
Y_train, Y_test = Y_true[:train_size], Y_true[train_size:]

# Create library (you can replace this with your custom library)
custom_library = ps.PolynomialLibrary(degree=7)

# Differentiation and optimizer
optimizer = ps.STLSQ(threshold=0.1)
model = ps.SINDy(feature_library=custom_library, optimizer=optimizer)

# Fit model on training set
dt = t[1] - t[0]
model.fit(Y_train, t=dt)

# Inspect discovered model
print("\n--- Learned Model ---")
model.print()

# --------------------------------
# Evaluate model
# --------------------------------
print("\n--- Evaluation ---")

# Evaluate on training data
train_score = model.score(Y_train, t=dt)
print(f"R² on training data: {train_score:.4f}")

# Evaluate on test data
test_score = model.score(Y_test, t=dt)
print(f"R² on test data: {test_score:.4f}")

# Evaluate on true (noise-free) data if available
if "Y_true" in globals():
    true_score = model.score(Y_true, t=dt)
    print(f"R² on true (noise-free) data: {true_score:.4f}")

index=-2
Y_sim = model.simulate(Y_true[index][0], t=t)

# --------------------------------------------
# Parameters
# --------------------------------------------
L = 1.0
skip = 10  # frame skipping for speed
scale_factor = 2.2 * L  # scaling for camera frame

# Assuming the arrays `Y_test`, `Y_sim`, and `t` already exist
# Replace these lines with your actual simulation data:
# ----------------------------------------------------------
# Example placeholders (remove in notebook and replace)
theta1_true = Y_test[index][:, 0]
theta2_true = Y_test[index][:, 1]
theta1_pred = Y_sim[:, 0]
theta2_pred = Y_sim[:, 1]
# ----------------------------------------------------------

# Convert angles to Cartesian coordinates
def to_cartesian(theta1, theta2, L=1.0):
    x1 = L * np.sin(theta1)
    y1 = -L * np.cos(theta1)
    x2 = x1 + L * np.sin(theta2)
    y2 = y1 - L * np.cos(theta2)
    return np.vstack([x1, y1, x2, y2]).T

# Example test data (for preview)
# Remove this section when running in your own environment
# n = 800
# t = np.linspace(0, 15, n)
# theta1_true = 0.8 * np.sin(t)
# theta2_true = 1.5 * np.sin(1.3*t)
# theta1_pred = theta1_true + 0.05*np.sin(0.7*t)
# theta2_pred = theta2_true + 0.05*np.cos(1.2*t)

true_cart = to_cartesian(theta1_true, theta2_true, L)
pred_cart = to_cartesian(theta1_pred, theta2_pred, L)

# --------------------------------------------
# Manim Scene
# --------------------------------------------
class DoublePendulumComparison(Scene):
    def construct(self):
        self.camera.background_color = "#0e0e0e"
        
        # Create pendulum elements
        origin = np.array([0, 0, 0])

        # True pendulum (blue)
        rod1_true = Line(origin, [true_cart[0,0], true_cart[0,1], 0], color=BLUE_B, stroke_width=6)
        rod2_true = Line(
            [true_cart[0,0], true_cart[0,1], 0],
            [true_cart[0,2], true_cart[0,3], 0],
            color=BLUE_B, stroke_width=6
        ).set_opacity(0.7)
        mass1_true = Dot(rod1_true.get_end(), color=BLUE_B)
        mass2_true = Dot(rod2_true.get_end(), color=BLUE_B)

        # Predicted pendulum (red)
        rod1_pred = Line(origin, [pred_cart[0,0], pred_cart[0,1], 0], color=RED_C, stroke_width=6)
        rod2_pred = Line(
            [pred_cart[0,0], pred_cart[0,1], 0],
            [pred_cart[0,2], pred_cart[0,3], 0],
            color=RED_C, stroke_width=6
        ).set_opacity(0.7)
        mass1_pred = Dot(rod1_pred.get_end(), color=RED_C)
        mass2_pred = Dot(rod2_pred.get_end(), color=RED_C)

        # Trails for end masses
        trace_true = TracedPath(mass2_true.get_center, stroke_color=BLUE_B, stroke_opacity=0.4, stroke_width=2)
        trace_pred = TracedPath(mass2_pred.get_center, stroke_color=RED_C, stroke_opacity=0.4, stroke_width=2)

        # Group all elements
        pendulum_group = VGroup(
            rod1_true, rod2_true, mass1_true, mass2_true,
            rod1_pred, rod2_pred, mass1_pred, mass2_pred,
            trace_true, trace_pred
        )

        self.add(pendulum_group)
        self.play(FadeIn(pendulum_group))
        self.wait(0.5)

        # --------------------------------------------
        # Animate time evolution
        # --------------------------------------------
        for i in range(0, len(t), skip):
            # True pendulum
            rod1_true.put_start_and_end_on(origin, [true_cart[i,0], true_cart[i,1], 0])
            rod2_true.put_start_and_end_on([true_cart[i,0], true_cart[i,1], 0], [true_cart[i,2], true_cart[i,3], 0])
            mass1_true.move_to(rod1_true.get_end())
            mass2_true.move_to(rod2_true.get_end())

            # Predicted pendulum
            rod1_pred.put_start_and_end_on(origin, [pred_cart[i,0], pred_cart[i,1], 0])
            rod2_pred.put_start_and_end_on([pred_cart[i,0], pred_cart[i,1], 0], [pred_cart[i,2], pred_cart[i,3], 0])
            mass1_pred.move_to(rod1_pred.get_end())
            mass2_pred.move_to(rod2_pred.get_end())

            self.wait(0.02)

        self.wait(1)
