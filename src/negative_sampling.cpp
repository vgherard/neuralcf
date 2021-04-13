#include <Rcpp.h>
#include <vector>
#include <unordered_set>
#include <utility>
using namespace Rcpp;

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
	std::vector<std::unordered_set<size_t> > positives(num_users);
	size_t user_id, item_id;
	for (size_t i = 0; i < num_positive; ++i) {
		user_id = user_input[i];
		item_id = item_input[i];
		positives[user_id - 1].insert(item_id);
	}
	for (size_t i = 0; i < num_positive; ++i) {
		user_id = user_input[i];
		item_id = item_input[i];
		size_t pos = i * (1 + num_negatives);
		user[pos] = user_id;
		item[pos] = item_id;
		label[pos] = 1;

		auto end_ptr = positives[user_id - 1].end();
		for (size_t k = 1; k <= num_negatives; ++k) {
			user[pos + k] = user_id;
			label[pos + k] = 0;
			while (true) {
				item_id = sample(num_items, 1)[0];
				auto ptr = positives[user_id - 1].find(item_id);
				if (ptr == end_ptr) {
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
