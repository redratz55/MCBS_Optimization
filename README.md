# MCBS_Optimization
An application of binary search to hyper-parameter optimization

This is the base implementation of binary search for hyper-parameter optimization.

Example:

```
from MCBS import mcm_opt_ucb
from sklearn.ensemble import RandomForestClassifier as RF

params = {'n_estimators': [1, 500], 'max_depth': [1, 50]}
restrictions = ['I', 'I']
mcbc_params, mcbc_score, mcbc_time = mcm_opt_ucb(x_train, y_train, 'accuracy', RF(random_state = seed), params, restrictions, seed = seed, I = 10, lhs_sampler=True)

````

The output of calling this method is the final optimized parameters, the best cross-validation score, and the total training time.

The algorithm works by creating a n-dimensional hyper-rectangle around the hyper-parameter space and repeatedly randomly sampling points within that space, recentering the rectangle, and shrinking every I iterations.

The restrictions parameter will be removed in future iterations, as the skopt.space method allows for sampling from the spaces without specifying the restrictions.



