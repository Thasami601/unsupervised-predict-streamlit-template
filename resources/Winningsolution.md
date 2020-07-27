This movie recommender app makes use of a content based recommender system to perform movie recommendations based on a user's 3 inputs. 

The recommendations were made based on the genres of the movies. Once the user has selected their 3 movies then 10 movies will be recommended to them based on similar genres. In order to ensure that the recommended list was weighted correctly we chose to make the recommendations based on the following split.

**Movie 1: 50% **

**Movie 2: 30% **

**Movie 3: 20% **


This ensures that the list has a good distribution covering various genres that could be inputted. The distance measure that was chosen is the cosine similarity. Cosine similarity is a metric used to measure how similar the documents are irrespective of their size and works well with a large sample size such as ours. 
