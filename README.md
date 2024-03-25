# Federated Human Mobility

**Definition of Client:** the venue visited by people.

**Setup:** The task is predicting how many people (optionally filtered by demographics such
as income range, age, gender, etc.) will visit a given venue category within the next x
hours/days. Nearby venues may experience a similar trend in popularity, e.g., if there is a
football match nearby, then all the food places may get more than the usual crowds.

**Input:** The input of our model must be a sequence of user check-ins for a specific venue.

**Architecture:** Some kind of RNN.

**Dataset:** Foursquare Dataset.

**Why FL?**
- The popularity can be a trade secret for the venues. Visits to certain places (like
hospitals) can be sensitive for people.
- Accuracy-wise, it’ll be better than training models at venues in isolation.

Train for all venues (venue category as input?) then query for each individual venue or
category depending on the inference needs. Think of it as simply predicting the number of
people coming to any specific venue, no matter the category.

This results in one client per venue, with multiple venues of multiple types.
Only one federated model is to be trained, on many different venue category types,
in order to predict its popularity.
