"""Different approaches. Let's restrict to the case of predicting one variable
from one neuron for now. Then go to case of predicting one variable from
multiple neurons. Maybe someday predict multiple variables.

0. U-test, auroc

1/2. No CV. Just score on the training set, but equalize across categories.

1. Leave-one-out CV. Fit to all but one sample. Predict that sample. Then
equalize over categories.

2. Like #1, but drop samples in the training set to equalize the categories.

3. Like #1, but hold out a pair of samples, one from each category, and then
score based on whethere this pair is correctly ordered. Main advantage is
that this should match U-test more closely.
"""

import numpy as np
import my
from ns5_process import myutils
#import scikits.learn as sklearn
import sklearn
import sklearn.linear_model
import matplotlib.pyplot as plt


def label_feature_with_integer(feature_vector, features=None):
    """Label the non-integer features with integers
    
    Example:
        feature_vector: ['le', 'ri', 'le', 'le']
        features: None
    
    Returns:
        output: [0, 1, 0, 0]
        features: ['le', 'ri']
    
    You can specify the intended ordering of the features. Otherwise, they
    the sorted unique values in the data.
    """
    if features is None:
        # Construct a mapping
        features = np.sort(np.unique(feature_vector))
    
    output = -np.ones(len(feature_vector), dtype=np.int)
    for n, feature in enumerate(features):
        output[feature_vector == feature] = n
    assert np.all(output >= 0)
    
    return output, features

def linreg_score2(df, output, input='count', rounding_tol=1e-8, **lm_kwargs):
    """Linearly regress `input` to predict `output`
    
    This fits a model of type LinearRegressions. Normally this is for
    predicting continuous variables, but we force the output of it to an 
    integer. Therefore your `output` must be integers as well.
    
    If it's within `tol` of a tie, issue warning (check that it's doing
    something reasonable in this case).
    
    Calculates performance for entire dataset, and also by equalizing over
    categories of `output`.
    
    One problem with this approach is that the model is highly affected
    by the empirical probabilities of each category. Basically it will tend
    very quickly toward always predicting the most-common category, unless
    there is strong evidence toward something smarter. Category-equalization
    makes it obvious that this is happening, but something more continuous
    might be a bit better ... (can make this happen with class_weight, somehow)
    
    Returns: dict with model and scores
    """
    # Coerce input to 2d array
    if input in df:
        input = df[input]
    input = np.asarray(input)
    if input.ndim == 1:
        input = input[:, None]
    
    # Coerce output to array
    if output in df:
        output = df[output]
    output = np.asarray(output)
    
    # Fit linear model
    lm = sklearn.linear_model.LinearRegression(**lm_kwargs)
    lm.fit(input, output)
    res = {'model': lm, 'n': len(input)}
    
    # Predict by rounding to int, and simple score
    output_pred = lm.predict(input)
    output_pred2 = np.rint(output_pred).astype(np.int)
    if np.any(np.abs(output_pred2 - output_pred) < rounding_tol):
        print "warning: close to tie in linreg_score2"
    res['flat_score'] = np.mean(output_pred2 == output)
    
    # Score by equalizing categories in `output`
    output_vals = np.unique(output)
    score_on_output = {}
    for output_val in output_vals:
        # Predict on this category
        input_subset = input[output == output_val]
        output_pred = lm.predict(input_subset)
        output_pred2 = np.rint(output_pred).astype(np.int)
        
        # Store performance on this output
        score_on_output[output_val] = np.mean(output_pred2 == output_val)
    
    # Overall score is score across categories
    res['equalized_score'] = np.mean(score_on_output.values())
    res['score_by_output'] = score_on_output
    
    return res

def logreg_score2(df, output, input='count', rounding_tol=1e-8, 
    auto_balance=True, cross_validate=False, **lm_kwargs):
    """Logistically regress `input` to predict `output`
    
    This fits a model of type LogisticRegression, which is categorical.
    
    auto_balance : if True, adjust the `class_weight` parameter so that
        each possible output of the discriminator is equally likely.
        If False, then the default behavior of LogisticRegression
        (`class_weight='auto'`) is used. This biases the discriminator to
        producing the most likely output. In noisy cases, the discriminator
        will always produce this output.
    
    There was a BUG in sklearn 0.12.1, and so I had to implement my own
    workaround for autobalance! With sklearn 0.14.1, this bug has been
    fixed, and you should now just always send class_weight='auto' to
    this function, which bypasses `auto_balance` completely.
    
    Calculates performance for entire dataset ("flat score"), and also by 
    equalizing over categories of `output` ("cateq_score").
    
    `flat` can award performance above chance even to stupid discriminators
    that always produce the most likely outcome, but `cateq` won't do this.
    Also, `flat` produces higher variance results, both higher and lower
    than `cateq`, for noisy data.
    
    `cateq_score` works pretty well regardless of the class_weight setting.
    It also tends to match AUROC pretty well, except in really noisy cases
    (where I think AUROC is upward biased) and in high signal cases 
    (where AUROC is higher because of what it measures).
    
    Returns: dict with model and scores
    """
    # Coerce input to 2d array
    if isinstance(input, basestring) and input in df:
        input = df[input]
    input = np.asarray(input)
    if input.ndim == 1:
        input = input[:, None]
    
    # Coerce output to array
    if isinstance(output, basestring) and output in df:
        output = df[output]
    output = np.asarray(output)
    
    # force equal probability output
    if 'class_weight' in lm_kwargs:
        auto_balance = False
    if auto_balance:
        output_vals = np.unique(output)
        class_weight = {}
        for output_val in output_vals:
            class_weight[output_val] = np.sum(output == output_val)
    
    
    if cross_validate:
        output_pred_l = []
        # Iterate over training sets
        for train_index, test_index in sklearn.cross_validation.LeaveOneOut(n=len(input)):
            # Create model object
            if auto_balance:
                lm = sklearn.linear_model.LogisticRegression(
                    class_weight=class_weight, 
                    **lm_kwargs)
            else:
                lm = sklearn.linear_model.LogisticRegression(**lm_kwargs)
            
            # Fit and predict
            lm.fit(input[train_index], output[train_index])
            output_pred_l.append(lm.predict(input[test_index]).item())

        output_pred = np.array(output_pred_l)
    else:
        # Create model object
        if auto_balance:
            lm = sklearn.linear_model.LogisticRegression(
                class_weight=class_weight, 
                **lm_kwargs)
        else:
            lm = sklearn.linear_model.LogisticRegression(**lm_kwargs)

        lm.fit(input, output)
        output_pred = lm.predict(input)

    res = {'model': lm, 'n': len(input)}

    
    # Simple score
    res['flat_score'] = np.mean(output_pred == output)
    
    # Score by equalizing categories in `output`
    output_vals = np.unique(output)
    score_on_output = {}
    for output_val in output_vals:
        # Predict on this category
        input_subset = input[output == output_val]
        #~ output_pred = lm.predict(input_subset)
        sub_output_pred = output_pred[output == output_val]
        
        # Store performance on this output
        score_on_output[output_val] = np.mean(sub_output_pred == output_val)
    
    # Overall score is score across categories
    res['equalized_score'] = np.mean(score_on_output.values())
    res['score_by_output'] = score_on_output
    
    return res

def linreg_score(input, output):
    """Linearly regress input and output, return score.
    
    input : (N_trials, N_features)
        Will be cast to 2d if necessary
    output : (N_trials,) must be 0s and 1s
    
    linear_model.LinearRegression is used to fit
    
    The score is calculated by making predictions on the input set
    (which is cheating). The predictions are thresholded at 0.5 and compared
    to the output.
    
    TODO: return jackknifed estimate of score instead

    Returns:
        score, fraction between 0 and 1
    """
    input, output = map(np.asarray, [input, output])
    
    # Input has to be 2d
    if input.ndim == 1:
        input = input[:, None]
    
    lm = sklearn.linear_model.LinearRegression()
    lm.fit(input, output)
    lm_score = np.sum((lm.predict(input) > .5).astype(np.int) == output)
    
    return lm_score / float(len(input))

def logreg_score(input, output):
    """Logisitically regress input and output, return score.
    
    input : (N_trials, N_features)
    output : (N_trials,) must be 0s and 1s
    
    linear_model.LogisticRegression is used to fit
    
    The score is calculated by making predictions on the input set
    (which is cheating). The predictions are then compared to the output.
    
    TODO: return jackknifed estimate of score instead

    Returns:
        score, fraction between 0 and 1
    """
    input, output = map(np.asarray, [input, output])
    
    # Input has to be 2d
    if input.ndim == 1:
        input = input[:, None]
    
    lm = sklearn.linear_model.LogisticRegression()
    lm.fit(input, output)
    lm_score = np.sum(lm.predict(input) == output)
    
    return lm_score / float(len(input))

def utest_score(input, output, take_max=False):
    """Returns AUROC between input and output
    
    input : (N_trials, N_features)
    output : (N_trials,) must be 0s and 1s
    take_max : whether to convert an auroc < .5 to 1-auroc
    
    The input is split based on the values in output. The AUROC between
    these groups is returned.
    
    Returns: AUROC, fraction between 0 and 1
    """
    input, output = map(np.asarray, [input, output])

    # Split trials by output variable
    masks = [output == val for val in [0, 1]]
    
    # Select input variable (spikes) by split
    x, y = [input[mask] for mask in masks]
    
    # Run AUROC
    #~ U, p, auroc = myutils.utest(x, y, return_auroc=True,
        #~ print_mannwhitneyu_warnings=print_mannwhitneyu_warnings, 
        #~ print_empty_data_warnings=print_empty_data_warnings)
    utest_res = my.stats.r_utest(x, y)
    
    if take_max and utest_res['auroc'] < .5:
        return 1 - utest_res['auroc']
    else:
        return utest_res['auroc']

def brute_force_score(input, output, n_steps=1000):
    """Returns score of best possible threshold (by trying all)
    
    input : (N_trials, N_features)
    output : (N_trials,) must be 0s and 1s
    
    Every threshold between input.min() and input.max() is tested and the
    best score is returned.
    
    Returns: score, fraction between 0 and 1
    """ 
    input, output = map(np.asarray, [input, output])

    # Split trials by output variable
    masks = [output == val for val in [0, 1]]
    
    # Select input variable (spikes) by split
    x, y = [input[mask] for mask in masks]    
    threshs = np.linspace(input.min(), input.max(), n_steps)
    scores = np.array([np.sum(y<th) + np.sum(x>=th) for th in threshs])
    
    # Also calculate inverse of score
    scores2 = len(output) - scores
    
    return np.max([scores, scores2]) / float(len(output))




