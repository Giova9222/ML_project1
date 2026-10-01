import numpy as np


def compute_MSEgrad(y, tx, w):
    """Compute the gradient of the mean squared error (MSE) with respect to w.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (n_samples,), containing the ground-truth labels.
    tx : np.ndarray
        Feature matrix of shape (n_samples, n_features), containing the input data.
    w : np.ndarray
        Weight vector of shape (n_features,), representing the current model parameters.

    Returns
    -------
    grad : np.ndarray
        Gradient of the MSE with respect to w, shape (n_features,).
    err : np.ndarray
        Residual vector y - tx @ w, shape (n_samples,).

    Example
    -------
    y = np.array([1.0, 2.0, 3.0])
    tx = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    w = np.array([0.0, 0.0])
    grad, err = compute_MSEgrad(y, tx, w)

    Example of use
    --------------
    grad, residuals = compute_MSEgrad(y_train, tx_train, current_w)
    updated_w = current_w - learning_rate * grad
    """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def compute_MSEloss(y, tx, w):
    """Return the mean squared error loss for targets y and predictions tx @ w.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (n_samples,).
    tx : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    w : np.ndarray
        Weight vector of shape (n_features,).

    Returns
    -------
    float
        Scalar MSE loss computed as sum((y - tx @ w)^2) / (2 * n_samples).

    Example
    -------
    y = np.array([1.0, 2.0, 3.0])
    tx = np.array([[1.0], [2.0], [3.0]])
    w = np.array([0.5])
    loss = compute_MSEloss(y, tx, w)

    Example of use
    --------------
    loss = compute_MSEloss(y_valid, tx_valid, weights)
    print(loss)
    """
    return np.sum((y - tx @ w) ** 2) / (len(y) * 2)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Train a linear model using gradient descent on the MSE objective.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (n_samples,).
    tx : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    initial_w : np.ndarray
        Initial weight vector of shape (n_features,).
    max_iters : int
        Maximum number of gradient descent iterations.
    gamma : float
        Learning rate used to update the weights.

    Returns
    -------
    w : np.ndarray
        Final optimized weight vector of shape (n_features,).
    final_loss : float
        Value of the MSE loss after the last update.

    Example
    -------
    y = np.array([1.0, 2.0, 3.0])
    tx = np.array([[1.0], [2.0], [3.0]])
    w0 = np.array([0.0])
    w, loss = mean_squared_error_gd(y, tx, w0, 100, 0.01)

    Example of use
    --------------
    weights, training_loss = mean_squared_error_gd(y_train, tx_train, w_init, 500, 0.1)
    """
    w = initial_w
    counter = 0
    while counter < max_iters:  # eventually implement early stopping by adding OR
        counter += 1
        grad, _ = compute_MSEgrad(y, tx, w)
        w = w - gamma * grad
    final_loss = compute_MSEloss(y, tx, w)
    return (w, final_loss)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """Yield minibatches of matching rows from y and tx, optionally in random order.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (n_samples,).
    tx : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    batch_size : int
        Number of samples to include per minibatch.
    num_batches : int, default=1
        Number of batches to generate in the iterator.
    shuffle : bool, default=True
        If True, batch starting indices are randomized; otherwise they follow a cyclic pattern.

    Yields
    ------
    tuple[np.ndarray, np.ndarray]
        A pair (y_batch, tx_batch) where y_batch has shape (batch_size,) and tx_batch has shape (batch_size, n_features).

    Example
    -------
    y = np.array([1, 2, 3, 4, 5])
    tx = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    for y_batch, tx_batch in batch_iter(y, tx, 2, num_batches=3, shuffle=False):
        print(y_batch, tx_batch)

    Example of use
    --------------
    for y_batch, tx_batch in batch_iter(y_train, tx_train, 32, num_batches=10):
        grad, _ = compute_MSEgrad(y_batch, tx_batch, current_w)
        current_w -= learning_rate * grad
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Train a linear model with stochastic gradient descent on the MSE objective.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (n_samples,).
    tx : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    initial_w : np.ndarray
        Initial weight vector of shape (n_features,).
    max_iters : int
        Number of SGD updates to perform.
    gamma : float
        Learning rate used for each weight update.

    Returns
    -------
    w : np.ndarray
        Final optimized weight vector of shape (n_features,).
    final_loss : float
        MSE value computed after training.

    Example
    -------
    y = np.array([1.0, 2.0, 3.0])
    tx = np.array([[1.0], [2.0], [3.0]])
    w0 = np.array([0.0])
    w, loss = mean_squared_error_sgd(y, tx, w0, 50, 0.01)

    Example of use
    --------------
    weights, loss = mean_squared_error_sgd(y_train, tx_train, w_init, 200, 0.05)
    """
    w = initial_w
    counter = 0
    while counter < max_iters:
        for y_batch, tx_batch in batch_iter(y, tx, 1, 1):
            counter += 1
            grad, _ = compute_MSEgrad(y_batch, tx_batch, w)
            w = w - gamma * grad
    final_loss = compute_MSEloss(y, tx, w)
    return (w, final_loss)


def least_squares(y, tx):
    """Fit a linear model by solving the least-squares system in closed form.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (n_samples,).
    tx : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    w : np.ndarray
        Optimal weight vector for the least-squares problem, shape (n_features,).
    final_loss : float
        MSE loss evaluated with the fitted weights.

    Example
    -------
    y = np.array([1.0, 2.0, 3.0])
    tx = np.array([[1.0], [2.0], [3.0]])
    w, loss = least_squares(y, tx)

    Example of use
    --------------
    weights, train_loss = least_squares(y_train, tx_train)
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    final_loss = compute_MSEloss(y, tx, w)
    return w, final_loss


def ridge_regression(y, tx, lambda_):
    """Fit a ridge-regularized linear model by solving the regularized least-squares system.

    Parameters
    ----------
    y : np.ndarray
        Target values of shape (n_samples,).
    tx : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    lambda_ : float
        Ridge regularization strength. Larger values increase the penalty on large weights.

    Returns
    -------
    w : np.ndarray
        Regularized weight vector of shape (n_features,).
    final_loss : float
        MSE loss computed using the ridge solution.

    Example
    -------
    y = np.array([1.0, 2.0, 3.0])
    tx = np.array([[1.0], [2.0], [3.0]])
    w, loss = ridge_regression(y, tx, 0.1)

    Example of use
    --------------
    weights, val_loss = ridge_regression(y_valid, tx_valid, lambda_value)
    """
    regfact = tx.shape[0] * 2 * lambda_ * np.eye(tx.shape[1])
    w = np.linalg.solve(tx.T @ tx + regfact, tx.T @ y)
    final_loss = compute_MSEloss(y, tx, w)
    return w, final_loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Train a logistic regression model using gradient descent.

    Parameters
    ----------
    y : np.ndarray
        Binary target values of shape (n_samples,), usually 0 or 1.
    tx : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    initial_w : np.ndarray
        Initial weight vector of shape (n_features,).
    max_iters : int
        Number of gradient descent iterations.
    gamma : float
        Learning rate used to update the weights.

    Returns
    -------
    w : np.ndarray
        Final learned weight vector of shape (n_features,).
    final_loss : float
        Logistic loss after the final iteration.

    Example
    -------
    y = np.array([0, 1, 1])
    tx = np.array([[0.2, 1.0], [1.0, 0.8], [1.5, 1.2]])
    w0 = np.array([0.0, 0.0])
    w, loss = logistic_regression(y, tx, w0, 200, 0.1)

    Example of use
    --------------
    weights, train_loss = logistic_regression(y_train, tx_train, w_init, 500, 0.01)
    """
    w = initial_w
    for i in range(max_iters):
        z = tx @ w
        sigma = np.exp(-np.logaddexp(0, -z))
        gradient = tx.T @ (sigma - y) / len(y)
        w = w - gamma * gradient
    z_final = tx @ w
    final_loss = np.sum(np.logaddexp(0, z_final) - y * z_final) / len(y)
    return (w, final_loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Train a regularized logistic regression model using gradient descent.

    Parameters
    ----------
    y : np.ndarray
        Binary target values of shape (n_samples,), usually 0 or 1.
    tx : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    lambda_ : float
        L2 regularization coefficient applied to the weights.
    initial_w : np.ndarray
        Initial weight vector of shape (n_features,).
    max_iters : int
        Number of gradient descent iterations.
    gamma : float
        Learning rate used to update the weights.

    Returns
    -------
    w : np.ndarray
        Final learned weight vector of shape (n_features,).
    final_loss : float
        Regularized logistic loss after the final iteration.

    Example
    -------
    y = np.array([0, 1, 1])
    tx = np.array([[0.2, 1.0], [1.0, 0.8], [1.5, 1.2]])
    w0 = np.array([0.0, 0.0])
    w, loss = reg_logistic_regression(y, tx, 0.01, w0, 200, 0.1)

    Example of use
    --------------
    weights, loss = reg_logistic_regression(y_train, tx_train, 0.1, w_init, 500, 0.01)
    """
    w = initial_w
    for i in range(max_iters):
        z = tx @ w
        sigma = np.exp(-np.logaddexp(0, -z))
        gradient = tx.T @ (sigma - y) / len(y) + 2 * lambda_ * w
        w = w - gamma * gradient
    z_final = tx @ w
    final_loss = np.sum(np.logaddexp(0, z_final) - y * z_final) / len(y)
    return (w, final_loss)
