#include <Rcpp.h>
#include <vector>
#include <unordered_set>
#include <utility>
using namespace Rcpp;

std::string make_key(size_t user_id, size_t item_id) {
	return std::to_string(user_id) + "_" + std::to_string(item_id);
}

// [[Rcpp::export]]
DataFrame add_negatives(
		IntegerVector user_input,
		IntegerVector item_input,
		size_t num_users,
		size_t num_items,
		size_t num_negatives
	)
{
	size_t num_positive = user_input.length();
	size_t n = (1 + num_negatives) * num_positive;
	IntegerVector user(n), item(n), label(n);

	// Build hash tables of positive items for each user
	// Fare in funzione separata chiamata una sola volta;
	// Sostituire vector<unordered_set<size_t>> con unordered_set<string>
	std::unordered_set<std::string> positives;
	size_t user_id, item_id;
	for (size_t i = 0; i < num_positive; ++i) {
		user_id = user_input[i];
		item_id = item_input[i];
		positives.insert( make_key(user_id, item_id) );
	}

	auto end_ptr = positives.end();
	for (size_t i = 0; i < num_positive; ++i) {
		user_id = user_input[i];
		item_id = item_input[i];
		size_t pos = i * (1 + num_negatives);
		user[pos] = user_id;
		item[pos] = item_id;
		label[pos] = 1;

		for (size_t k = 1; k <= num_negatives; ++k) {
			user[pos + k] = user_id;
			label[pos + k] = 0;
			while (true) {
				item_id = sample(num_items, 1)[0];
				std::string key = make_key(user_id, item_id);
				if (positives.find(key) == end_ptr) {
					item[pos + k] = item_id;
					break;
				}
			}
		}
	}

	DataFrame res  = DataFrame::create(
		Named("user") = user,
		Named("item") = item,
		Named("label") = label
	);
	return res;
}
