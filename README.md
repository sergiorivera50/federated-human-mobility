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

**How to run this?**

Follow `TEMPLATE.md` instructions on how to setup the project template.

Then, run the prepare dataset to download and partition the Foursquare dataset.

```bash
poetry run python -m project.task.human_mobility.dataset_preparation
```

After, you can modify the configuration task to perform several experiments, then simply run the configuration.

```bash
poetry run python -m project.main --config-name=human_mobility
```
