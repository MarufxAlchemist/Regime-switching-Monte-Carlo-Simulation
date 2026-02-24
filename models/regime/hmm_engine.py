import numpy as np
from hmmlearn import hmm

# Number of assets
NUM_ASSETS = 10  

# Define true parameters for HMM
TRUE_A = [
    [0, 0.5, 0.5],
    [0.6, 0, 0.4],
    [0.3, 0.3, 0.4]
]
TRUE_MU = [
    np.array([0.1, -0.2, 0.1])[:NUM_ASSETS],   # returns for NUM_ASSETS assets
    np.array([-0.2, 0.3, -0.4])[:NUM_ASSETS],   # returns for NUM_ASSETS assets
    np.array([0.1, 0.1, 0.2])[:NUM_ASSETS]        # returns for NUM_ASSETS assets
]
TRUE_SIGMA = [
    np.eye(NUM_ASSETS) * 0.03,      # vols for NUM_ASSETS assets
    np.eye(NUM_ASSETS) * 0.05,       # vols for NUM_ASSETS assets
    np.eye(NUM_ASSETS) * 0.1          # vols for NUM_ASSETS assets
]
TRUE_PI = [0.3, 0.4, 0.3]   # initial state probabilities

# Define the number of states and generate observed data
NUM_STATES = 3   
T = 250            # sequence length
OBSERVATIONS = np.empty((T, NUM_ASSETS))
STATES = []

np.random.seed(1987)  # for reproducibility
for t in range(T):
    s = STATES[-1] if STATES else np.random.choice([0, 1, 2], p=TRUE_PI)   # initial state
    STATES.append(s)
    OBSERVATIONS[t] = np.random.multivariate_normal(TRUE_MU[s], TRUE_SIGMA[s])

# Fit a Gaussian HMM with the observed data
model = hmm.GaussianHMM(n_components=NUM_STATES, covariance_type="full")
model.startprob_ = np.array([0.6, 0.3, 0.1])   # start probabilities
model.transmat_ = TRUE_A                         # transition matrix
model.fit(OBSERVATIONS)

# Predict the most likely sequence of states (i.e., argmax over states)
logprob, alphas = model.decode(OBSERVATIONS, algorithm="viterbi")
pred_states = np.argmax(alphas, 1)   # convert from state indices to state probabilities

# Calculate accuracy and confusion matrix for state prediction analysis
confusion = np.zeros((NUM_STATES, NUM_STATES))
for true_state, pred_state in zip(STATES, pred_states):
    confusion[true_state, pred_state] += 1
accuracy = np.sum(np.diag(confusion)) / np.sum(confusion)
print(f"State prediction accuracy: {accuracy * 100:.2f}%")
