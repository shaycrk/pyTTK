### Biased Penalty TTK
This version of the TTK script allows for biased penalties to reflect different costs of false positive and false negatives. Specifically, for precision@k, we care much more about negative examples above the decision boundary than about positive examples below it (if we care about the latter at all), so we might run with a much larger value of Cn than Cp, or even with Cp=0

It should, however, be considered an experimental work in progress. In particular, because TTK is optimizing for at most k examples predicted positive (rather than exactly k), introducing biased penalties for false negatives vs positives seems to have a tendency to yield results with only a handful (or even no) cases predicted in the positive class.

