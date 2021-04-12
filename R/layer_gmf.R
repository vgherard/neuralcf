GMFLayer <- R6::R6Class("GMFLayer",

	inherit = keras::KerasLayer,

	private = list(

		# Layer defining variables
		n = NULL,
		emb_dim = NULL,
		out_activation = NULL,
		emb_l2_penalty = NULL,
		out_l2_penalty = NULL,

		# Layer weights
		emb = list(),
		dot_diag_metric = NULL,

		# Utilities
		emb_l2_reg = function(i) {
			.regularizer_l2(private$emb_l2_penalty[[i]])
		},
		out_l2_reg = function() {
			.regularizer_l2(private$out_l2_penalty)
		},

		emb_shape = function(i) {
			list(private$n[[i]] + 1L, private$emb_dim)
		},
		diag_metric_shape = function() {
			list(private$emb_dim, 1L)
		},

		emb_lookup = function(x) {
			vec <- list()
			tf_lookup <- tensorflow::tf$nn$embedding_lookup
			for (i in 1:2)
				vec[[i]] <- tf_lookup(private$emb[[i]], x[[i]])
			return(vec)
		}

	),
	active = ,

	public = list(
		initialize = function(n,
				      emb_dim,
				      out_activation,
				      emb_l2_penalty,
				      out_l2_penalty
				      )
		{
			private$n <- as.integer(n)
			private$emb_dim <- as.integer(emb_dim)
			private$out_activation <- out_activation
			private$emb_l2_penalty <- emb_l2_penalty
			private$out_l2_penalty <- out_l2_penalty
		},

		build = function(input_shape)
		{
			private$emb <- lapply(1:2, function(i) {
				self$add_weight(
					name = paste("emb", i, sep = "_"),
					shape = private$emb_shape(i),
					initializer = "glorot_uniform",
					regularizer = private$emb_l2_reg(i),
					trainable = TRUE
					)
			})

			private$dot_diag_metric <- self$add_weight(
				name = "dot_diag_metric",
				shape = private$diag_metric_shape(),
				initializer = "glorot_uniform",
				regularizer = private$out_l2_reg(),
				trainable = TRUE
				)
		},

		call = function(x, mask = NULL) {
			vec <- private$emb_lookup(x)
			prod <- tensorflow::tf$multiply(vec[[1]], vec[[2]])
			k_squeeze(
				k_dot(prod, private$dot_diag_metric),
				axis = -1
				)
		},

		compute_output_shape = function(input_shape) {
			reticulate::tuple(input_shape[[1]])
		}

		#,print = function(...){}
	)
)

#' @export
layer_gmf <- function(
	object,
	n,
	emb_dim,
	out_activation = NULL,
	emb_l2_penalty = NULL,
	out_l2_penalty = NULL,
	name = NULL,
	trainable = TRUE
	)
{
	keras::create_layer(
		GMFLayer, object, list(n = n,
				       emb_dim = emb_dim,
				       out_activation = out_activation,
				       emb_l2_penalty = emb_l2_penalty,
				       out_l2_penalty = out_l2_penalty,
				       name = name,
				       trainable = trainable
				       )
		)
}
