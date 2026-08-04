# EB SMOTE for continuous and nominal data

- pointdef_NC: The definition of the position for an generated synthetic instance. First for the continuos part then for the nominal features.
- enh_border_NC: Calculation of relevant nearest neighbour => call pointdef_NC (one synthetic instance for each minority neighbour)
- enh_border_level_NC: Same as enh_border_NC. But in a loop until the aimed balance level is reached.
- balancing_NC: Selection of Sampling Method      
- pipeline3_NC: The pipeline for the different methods (load, clean, preprocess, balance, solve, calculate measure)
