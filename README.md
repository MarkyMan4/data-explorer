# data-explorer
This app will let you import your own data set and explore it with various visualizations and regression models

# Todo
- [x] lots of graphs
- [x] multiple options for types of regression
- [x] select existing file, detect columns
- [ ] input validation
- [ ] compare types of regression and recommend a model based on mean squared error
- [ ] file upload - feature request exists so devs are aware: https://github.com/streamlit/streamlit/issues/120
- [ ] improve UI
- [ ] rather than just predicting a single value, split data into testing and training sets and show MAE
- [ ] possibly incorporate neural networks, kinda expensive computations for my machine though
- [ ] more options for models
- [ ] instead of always doing a degree 15 polynomial, test degrees 1 - 15 and choose the one with the lowest MAE
- [ ] allow for users to join datasets - auto detect columns that can be joined on

# Issues
- none for now

# Resolved issues
- Transformations don't show up on files until the server has been restarted. This is because I'm caching the data when it first gets read from the datasource. It doesn't detect any changes to the file because of this. There might be a good way to work around this - possibly a way to clear the cache and have it reload after a transformation has been made. Less preferably, I could avoid caching transformed data files, but this could cause a hit to performance on large data sets.
