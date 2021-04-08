GMFLayer <- R6::R6Class("GMFLayer",

	inherit = keras::KerasLayer,

	public = list(
		emb_dim = NULL,
		num_users = NULL,
		num_items = NULL,

		user_emb = NULL,
		item_emb = NULL,
		metric = NULL,

		initialize = function(emb_dim, num_users, num_items)
		{
			self$emb_dim <- as.integer(emb_dim)
			self$num_users <- as.integer(num_users)
			self$num_items <- as.integer(num_items)
		},

		build = function(input_shape)
		{
			self$user_emb <- self$add_weight(
				name = "user_emb",
				shape = list(self$num_users + 1L, self$emb_dim),
				initializer = initializer_random_normal(),
				trainable = TRUE
				)

			self$item_emb <- self$add_weight(
				name = "item_emb",
				shape = list(self$num_items + 1L, self$emb_dim),
				initializer = initializer_random_normal(),
				trainable = TRUE
				)

			self$metric <- self$add_weight(
				name = "metric",
				shape = list(self$emb_dim, 1L),
				initializer = initializer_random_normal(),
				trainable = TRUE
				)
		},

		call = function(x, mask = NULL) {
			user_vec <- tensorflow::tf$nn$embedding_lookup(
				params = self$user_emb, ids = x[[1]]
				)
			item_vec <- tensorflow::tf$nn$embedding_lookup(
				params = self$item_emb, ids = x[[2]]
			)
			x <- tensorflow::tf$multiply(user_vec, item_vec)
			k_squeeze(
				k_dot(x, self$metric),
				axis = -1
				)
		},

		compute_output_shape = function(input_shape) {
			reticulate::tuple(input_shape[[1]])
		}
	)
)

#' @export
layer_gmf <- function(
	object, emb_dim, num_users, num_items, name = NULL, trainable = TRUE
	)
{
	keras::create_layer(
		GMFLayer, object, list(emb_dim = as.integer(emb_dim),
				       num_users = as.integer(num_users),
				       num_items = as.integer(num_items),
				       name = name,
				       trainable = trainable
				       )
		)
}
