GMFLayer <- R6::R6Class("GMFLayer",

	inherit = keras::KerasLayer,

	private = list(

		# Layer defining variables
		n = NULL,
		emb_dim = NULL,
		emb_l2_penalty = NULL,

		# Layer weights
		emb = list(),

		# Utilities
		emb_l2_reg = function(i) {
			.regularizer_l2(private$emb_l2_penalty[[i]])
		},
		emb_shape = function(i) {
			list(private$n[[i]] + 1L, private$emb_dim)
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
		initialize = function(n, emb_dim, emb_l2_penalty)
		{
			private$n <- as.integer(n)
			private$emb_dim <- as.integer(emb_dim)
			private$emb_l2_penalty <- emb_l2_penalty
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
		},

		call = function(x, mask = NULL) {
			vec <- private$emb_lookup(x)
			k_squeeze(
				tensorflow::tf$multiply(vec[[1]], vec[[2]]),
				axis = 2
				)
		},

		compute_output_shape = function(input_shape) {
			reticulate::tuple(input_shape[[1]], private$emb_dim)
		}

		#,print = function(...){}
	)
)

#' @export
layer_gmf <- function(
	object,
	n,
	emb_dim,
	emb_l2_penalty = NULL,
	name = NULL,
	trainable = TRUE
	)
{
	check_args_gmf(n, emb_dim, emb_l2_penalty)
	c(emb_l2_penalty) %<-% vectorize_args(emb_l2_penalty, len = 2)
	keras::create_layer(
		GMFLayer, object, list(n = n,
				       emb_dim = emb_dim,
				       emb_l2_penalty = emb_l2_penalty,
				       name = name,
				       trainable = trainable)
		)
}

check_args_gmf <- function(n, emb_dim, emb_l2_penalty)
{
	tryCatch(
		# try
		assertthat::assert_that(
			is_n_inputs(n),
			is_emb_dim(emb_dim, len = 1),
			is_l2_penalty(emb_l2_penalty, len = 2)
		)
		,
		# catch
		error = function(cnd)
			rlang::abort(cnd$message, class = "domain_error")
	)
}
