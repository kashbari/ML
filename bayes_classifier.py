# Implement Naives Bayes Classification

def bayes(model,test):
	predictions = list()
	for row in test:
		output = predict(model,row)
		predictions.append(output)
	return predictions


def predict(summaries,row):
	probabilities = calculate_class_prob(summaries,row)
	best_label, best_prob = None,-1
	for class_value, probability in probabilities.items():
		if best_label is None or probabilitiy > best_prb:
			best_prob = probability
			best_label = class_value
	return best_label

# fit model
# model = summarize_by_class(dataset)

#predict the label
# label = predict(model,row)
