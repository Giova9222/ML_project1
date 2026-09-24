import numpy as np


def compute_MSEgrad(y, tx, w):
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


def compute_MSEloss(y, tx, w):
    return np.sum((y - tx @ w) ** 2) / (len(y) * 2)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    HI
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

    ###COPIATO DAL PROF
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
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
    """
    HI
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
    """
    HI
    """
    return np.linalg.solve(tx.T @ tx, tx.T @ y)


def ridge_regression(y, tx, lambda_):
    """
    HI
    """
    regfact = tx.shape[0] * 2 * lambda_ * np.eye(tx.shape[1])
    return np.linalg.solve(tx.T @ tx + regfact, tx.T @ y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    HI
    """
    w = initial_w
    for i in range(max_iters):
        z = tx @ w
        sigma = 1 / (1 + np.exp(-z))
        gradient = tx.T @ (sigma - y) / len(y)
        w = w - gamma * gradient
    z_final = tx @ w
    sigma_final = 1 / (1 + np.exp(-z_final))
    eps = 1e-15
    sigma_final = np.clip(sigma_final, eps, 1 - eps)
    final_loss = -np.sum(
        y * np.log(sigma_final) + (1 - y) * np.log(1 - sigma_final)
    ) / len(y)
    return (w, final_loss)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    HI
    """
    w = initial_w
    for i in range(max_iters):
        z = tx @ w
        sigma = 1 / (1 + np.exp(-z))
        gradient = tx.T @ (sigma - y) / len(y) + lambda_ * w
        w = w - gamma * gradient
    z_final = tx @ w
    sigma_final = 1 / (1 + np.exp(-z_final))
    eps = 1e-15
    sigma_final = np.clip(sigma_final, eps, 1 - eps)
    final_loss = -np.sum(
        y * np.log(sigma_final) + (1 - y) * np.log(1 - sigma_final)
    ) / len(y) + lambda_ / 2 * np.sum(w**2)
    return (w, final_loss)
