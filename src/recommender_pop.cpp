#include <Rcpp.h>
#include <vector>
using namespace Rcpp;
using std::vector;

// [[Rcpp::export]]
NumericVector compute_average_ratings(IntegerVector ids, NumericVector ratings)
{
	size_t n_ids = max(ids);

	NumericVector res(n_ids + 1);

	vector<double> rating_sums(n_ids + 1);
	vector<int> counts(n_ids + 1);
	double sum_all = 0;

	size_t id;
	double rating;
	size_t len = ids.length();
	for (size_t j = 0; j < len; ++j) {
		id = ids[j];
		rating = ratings[j];
		rating_sums[id - 1] += rating;
		sum_all += rating;
		++counts[id - 1];
	}

	double avg_rate = sum_all / ratings.length();
	res[n_ids] = avg_rate;
	for (size_t i = 0; i < n_ids; ++i) {
		if (counts[i] == 0)
			res[i] = avg_rate;
		else
			res[i] = rating_sums[i] / counts[i];
	}

	return res;
}

// [[Rcpp::export]]
NumericVector compute_average_interactions(IntegerVector ids, int num_objects)
{
	size_t n_ids = max(ids);

	NumericVector res(n_ids + 1);

	vector<int> counts(n_ids + 1);

	size_t id;
	size_t len = ids.length();
	for (size_t j = 0; j < len; ++j)
		++counts[ids[j] - 1];

	for (size_t i = 0; i < n_ids; ++id)
		res[i] = counts[i] / num_objects;
	res[n_ids] = 0;

	return res;
}

// [[Rcpp::export]]
NumericVector return_average_prediction(
		IntegerVector ids, NumericVector preds, int num_ids
	)
{
	size_t len = ids.length();
	double generic_pred = preds[preds.length() - 1];
	NumericVector res(len);
	for (size_t j = 0; j < len; ++j) {
		res[j] = ids[j] <= num_ids ? preds[ids[j]] : generic_pred;
	}
	return res;
}
