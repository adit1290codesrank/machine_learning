import numpy as np
from sklearn.linear_model import SGDRegressor
import time

# 1. SETUP DATA (Matches your C++ Main)
m = 10000  # Samples
n = 500    # Features
# Generate Random Data
X = np.random.rand(m, n)
true_weights = np.random.rand(n)
noise = np.random.rand(m) - 0.5
y = X.dot(true_weights) + 5 + noise # y = X*w + bias + noise

# 2. SETUP MODEL
# SGDRegressor is the exact equivalent of your Gradient Descent loop
# We turn off "early stopping" to force it to run the full iterations like yours
model = SGDRegressor(
    learning_rate='constant', 
    eta0=0.01, 
    max_iter=1000, 
    tol=None,           # Don't stop early
    shuffle=False,      # Don't shuffle (to be identical to yours)
    penalty=None        # No regularization (L2 is default in sklearn)
)

# 3. RUN BENCHMARK
start_time = time.time()
model.fit(X, y)
end_time = time.time()

print(f"Training Time: {end_time - start_time:.4f} seconds")