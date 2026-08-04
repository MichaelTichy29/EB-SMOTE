# EB SMOTE for continuous data

- pointdef: The definition of the position for an generated synthetic instance
- enh_border: Calculation of relevant nearest neighbour => call pointdef (one synthetic instance for each minority neighbour)
- enh_border_level: Same as enh_border. But in a loop until the aimed balance level is reached.
- balancing: Selection of Sampling Method     
- pipeline3: The pipeline for the different methods (load, clean, preprocess, balance, solve, calculate measure)
