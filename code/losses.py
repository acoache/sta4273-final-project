import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

"""
# Loss functions

* EnsembleWithPrior_Loss -- Function to train the weights/biases of an ensemble of neural nets
* Ensemble_GaussianParamNet_2D_Loss -- Function to train the weights/biases of an ensemble of GaussianParamNets
* Ensemble_GaussianParamNet_2D_Sample -- Function to sample from output of an ensemble of GaussianParamNets
* Weight_Unif_Loss -- Function to keep weights of the ensembe unchanged
* Weight_PG_Loss -- Function to train weights of the ensemble using a policy gradient approach
* Weight_Exp3_Loss -- Function to train weights of the ensemble using a bandit approach
"""

def EnsembleWithPrior_Loss(data, predictor, oracle, discount, optimizer, num_actions):
    
    # format batch of observations
    current_states = np.asarray(data[0])
    actions = np.asarray(data[1])
    rewards = np.asarray(data[2])
    continue_play_inds = np.asarray(data[3])
    new_states = np.asarray(data[4])
    
    # target rewards from the oracle
    target_qs = oracle(np.atleast_2d(new_states).astype('float32'))
    max_targets = np.amax(target_qs, axis=-1)
    target = rewards[:, np.newaxis] + discount * continue_play_inds[:, np.newaxis] * max_targets

    # calculate prediction from the predictor
    with tf.GradientTape() as tape:
        one_hot_actions = tf.one_hot(actions, num_actions)
        main_predictions = predictor(np.atleast_2d(current_states).astype('float32'))
        pred = tf.einsum('bka,ba->bk', main_predictions, one_hot_actions)
        loss = tf.square(pred - target)
    
    # optimizer step only for used variables
    variables = predictor.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        (grad, var) 
        for (grad, var) in zip(gradients, variables) 
        if grad is not None
    )

# note this ignores the num_actions param
# currently uses L2 regularization, can be tweaked
# a nice example of deepQN loss that I want to save to reference later
# https://stats.stackexchange.com/questions/249355/how-exactly-to-compute-deep-q-learning-loss-function
# slightly problematic for this type of Q value estimation -- could be remedied by doing diagonal Gaussian 
# my strategy is to estimate a Q value for each action, even the ones not taken, by setting their reward to 0
def Ensemble_GaussianParamNet_2D_Loss(data, predictor, oracle, discount, optimizer, num_actions,
                                     regularizer_lr=0.01):
    
    # extract data
    current_states = np.asarray(data[0])
    actions = np.asarray(data[1])
    rewards = np.asarray(data[2]).astype('float32')
    continue_play_inds = np.asarray(data[3]).astype('float32')
    new_states = np.asarray(data[4])
    
    # copy the observations across ensembles and actions
    one_hot_actions = tf.expand_dims(tf.one_hot(actions, num_actions, dtype='float32'),axis=1) 
    rewards = tf.expand_dims(tf.repeat(rewards[:,np.newaxis], repeats=[2],axis=1),axis=1)
    continue_play_inds = tf.expand_dims(tf.repeat(continue_play_inds[:,np.newaxis], repeats=[2],axis=1),axis=1)

    # compute the best action using the mean of the target
    target_q_mu = oracle(np.atleast_2d(new_states).astype('float32'))[0]
    max_mu = np.amax(target_q_mu, axis=-1)[:, :, np.newaxis]

    # loop over each ensemble, needs improvement if ensemble size is large
    targets = []
    for idx in range(max_mu.shape[1]):

        max_mu_idx = max_mu[:,idx,:]
        max_mu_idx = tf.expand_dims(tf.repeat(max_mu_idx, repeats=[2],axis=1),axis=1)
        targets.append(one_hot_actions * rewards + discount * continue_play_inds * max_mu_idx)

    target = tf.stack(targets,axis=1) # this is like (data, ensemble, 1, action)

    with tf.GradientTape() as tape:
        
        # predictor network parametrizes Gaussian over q values
        main_mu, main_diag, main_odiag = predictor(np.atleast_2d(current_states).astype('float32'))

        main_mu = tf.expand_dims(main_mu, axis=2)
        main_mu_T = tf.transpose(main_mu,perm=[0,1,3,2])

        # some helpers to extract the Cholesky in a differentiable way
        diag_helper = tf.expand_dims(tf.expand_dims(tf.constant([[1,0],[0,1]], dtype='float32'),axis=0),axis=0)
        main_diag_mat = tf.expand_dims(main_diag,axis=2) * diag_helper

        odiag_helper = tf.expand_dims(tf.expand_dims(tf.constant([[0,0],[1,0]], dtype='float32'),axis=0),axis=0)
        main_odiag_mat = tf.expand_dims(main_odiag,axis=2) * odiag_helper

        main_cholesky = main_diag_mat + main_odiag_mat
        main_cholesky_T = tf.transpose(main_cholesky, perm=[0,1,3,2])

        # log of regularized MLE objective

        # compute log(det(L)) with clipped values to avoid nans
        log_chol_det = tf.math.log(tf.clip_by_value(tf.math.reduce_prod(main_diag,axis=-1,keepdims=True),0.01,10000))

        # compute (target - mu)' L L' (target - mu)
        main_mu_target_diff_T = target - main_mu
        main_mu_target_diff = tf.transpose(main_mu_target_diff_T, perm=[0,1,3,2])
        loss_prod_lhs = tf.matmul(main_mu_target_diff_T,main_cholesky)
        loss_prod_rhs = tf.matmul(main_cholesky_T,main_mu_target_diff)
        loss_prod = tf.reduce_prod(tf.matmul(loss_prod_lhs,loss_prod_rhs),axis=-1)

        # - log(det(L)) + (1/2) (target - mu)' L L' (target - mu) + mu'mu
        # sum over data
        loss = tf.reduce_sum(-log_chol_det + 0.5*loss_prod + regularizer_lr * tf.sqrt(tf.reduce_prod(tf.matmul(main_mu,main_mu_T),axis=-1)),axis=0)
    
    # optimizer step
    variables = predictor.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(
        (grad, var) 
        for (grad, var) in zip(gradients, variables) 
        if grad is not None
    )

# samples using precision matrix
def Ensemble_GaussianParamNet_2D_Sample(param_tuple):
    mu, diag, odiag = param_tuple[0], param_tuple[1], param_tuple[2]

    dd = np.diag([1.,1.])
    od = np.zeros((2,2))
    od[1,0] = 1.

    samples = []

    # loop over each ensemble, needs improvement if ensemble size is large
    for idx in range(mu.shape[1]):

        mu_idx = mu[:,idx,:].numpy()
        diag_idx = diag[:,idx,:].numpy()
        odiag_idx = odiag[:,idx,:].numpy()

        chol_idx = diag_idx * dd + odiag_idx * od
        chol_T_idx = np.transpose(chol_idx)

        Z = np.random.multivariate_normal(mean=np.zeros(2), cov=np.diag([1.,1.])) # standard normal
        X_idx = np.matrix(np.linalg.solve(chol_T_idx, Z)) # faster than inv

        samples.append(tf.Variable(X_idx + mu_idx))

    return tf.stack(samples,axis=1)

def Weight_Unif_Loss(weights, active_nets, rewards, weight_memory, optimizer):
    
    return weights

def Weight_PG_Loss(weights, active_nets, rewards, weight_memory, optimizer):
    
    # calculate the objective using the policy gradient theorem
    with tf.GradientTape() as tape:
        log_prob = weights.log_prob(active_nets)
        weights_loss = -tf.reduce_mean(log_prob * (rewards-tf.reduce_mean(rewards)))
    
    # optimizer step
    variables = weights.trainable_variables
    gradients = tape.gradient(weights_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return weights

def Weight_Exp3_Loss(weights, active_nets, rewards, weight_memory, optimizer, exp3_lr, exp3_eps=0):

    current_weights = weights.probs_parameter().numpy()
    num_ensemble = len(current_weights)
    old_probs = np.array(weight_memory)

    # build one hot matrix
    arms = np.array(active_nets)
    arms_mat = np.zeros((arms.size, num_ensemble))
    arms_mat[np.arange(arms.size),arms] = 1
    
    # Exp3 algorithm
    losses = -np.array(rewards).reshape((len(rewards),1))
    iw_losses = arms_mat * losses / old_probs
    unnorm_weights = np.exp(-exp3_lr * np.sum(iw_losses, axis=0))
    new_weights = tfp.distributions.Categorical(probs = tf.Variable(unnorm_weights / sum(unnorm_weights)))
    
    # compute smoothed weights
    uniform_weights = tfp.distributions.Categorical(probs = tf.Variable(np.ones(num_ensemble) / num_ensemble))
    smoothed_weights = tfp.distributions.Categorical(probs = tf.Variable(exp3_eps*uniform_weights.probs + (1-exp3_eps)*new_weights.probs))
    
    return smoothed_weights

