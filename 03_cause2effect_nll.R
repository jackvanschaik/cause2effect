# this was run on seperate server, with GPUs and CUDA
# Read Data ####################################################################
library(tidyverse)
library(torch)
cn_count <- read_csv("data_out/cause_net_1M.csv")

if (cuda_is_available()) {
    dev <- torch_device("cuda")
} else {
    dev <- torch_device("cpu")
}

# Word-index map
vocab <- sort(unique(c(cn_count$cause, cn_count$effect)))
cn_count$cause <- as.integer(factor(cn_count$cause, levels=vocab))
cn_count$effect <- as.integer(factor(cn_count$effect, levels=vocab))

# Transform to torch-ready format
L_s <- lapply(1:nrow(cn_count), function(i) setNames(as.list(cn_count[i,]), c("x", "y")))

# Load Data Into Torch #########################################################
cn_dataset <- dataset(
    name = "causenet_ae",

    initialize = function(L, vocab) {
        self$L <- lapply(L, function(z) {
            list(
                x = torch_tensor(as.integer(z$x), device = dev),
                y = torch_tensor(as.integer(z$y), device = dev)
            )
        })
        self$vocab <- vocab
    },

    .getitem = function(i) {
        self$L[[i]]
    },

    .length = function() {
        length(self$L)
    }
)

# Create dataset and dataloaders
train_ds <- cn_dataset(L_s, vocab) # takes 10-15 minutes
train_dl <- dataloader(train_ds, batch_size = 32, shuffle = TRUE)

# Create Model Architecture ####################################################
autoenc_model <- nn_module(
    initialize = function(emb_dim, v) {
        self$emb_layer <- nn_embedding(v, emb_dim)
        self$lin_layer <- nn_linear(emb_dim, v)
    },

    forward = function(x) {
        x %>%
            self$emb_layer() %>%
            nnf_sigmoid() %>%
            self$lin_layer() ->
            y

        y

    }
)

# Train The Model ##############################################################

# Initialization
n_epc <- 5
if (cuda_is_available()) {
    model <- autoenc_model(50, length(vocab))$cuda()
} else {
    model <- autoenc_model(50, length(vocab))
}
optimizer <- optim_adam(model$parameters, lr = 0.001)

# Train
for (ep in 1:n_epc) {
    model$train()
    train_losses <- c()

    k <- 0
    N <- length(train_dl)

    coro::loop(for (b in train_dl) {
        k <<- k + 1

        if (k %% 100 == 0) {
            print(paste0("Epoch ", ep, ": ", k/N))
        }

        optimizer$zero_grad()
        output <- model(b$x)$squeeze(2)
        loss <- nnf_cross_entropy(output, torch_squeeze(b$y, 2), reduction = "mean")

        loss$backward()
        optimizer$step()

        train_losses <- c(train_losses, loss$item())

    })
}

# Save The Embeddings ##########################################################
C_ <- model$emb_layer
cause_emb <- matrix(NA, nrow = length(vocab), ncol = 50)
for (j in 1:length(vocab)) {
    t_in <- torch_tensor(as.integer(j), device=dev)
    t_out <- C_$forward(t_in)$cpu()
    cause_emb[j,] <- as.numeric(t_out)
}
effect_emb <- as.matrix(model$lin_layer$weight$cpu())

cbind(
    data.frame(token = vocab),
    cause_emb,
    effect_emb
) %>%
    setNames(c("token", paste0("c", 1:50), paste0("e", 1:50))) ->
    ce_emb

## Export Embeddings
saveRDS(ce_emb, "embeddings/cn_ce_100_2.rds")
readr::write_csv(ce_emb, "embeddings/cn_ce_100_2.csv")
saveRDS(
    list(
        M_c = cause_emb,
        M_e = effect_emb
    ),
    "matrices/cn_ce_100_2.rds"
)


# The following evaluation isn't used in the final paper, file 05 is used
# it should be the same-- but the evaluation file has all scores in one place
# Load Evaluation Set ##########################################################
eval <- function(pred, actl) {
    tp <- sum((actl == 1) & (pred == 1))
    tn <- sum((actl == 0) & (pred == 0))
    fp <- sum((actl == 0) & (pred == 1))
    fn <- sum((actl == 1) & (pred == 0))
    prec <- tp/(tp + fp)
    rcll <- tp/(tp + fn)
    accu <- (tp + tn)/(tp + tn + fp + fn)
    f1   <- 2*tp/(2*tp + fp + fn)

    data.frame(f1 = f1, precision = prec, recall = rcll, accuracy = accu)
}

cn_count <- read_csv("data_out/causenet_with_counts.csv")

# Get the vocabulary of unique tokens
vocab <- sort(unique(c(cn_count$cause, cn_count$effect)))

#First, get a representative sample
M <- matrix(NA_character_, nrow = 2, ncol = sum(cn_count$count))
j <- 1
for (i in 1:nrow(cn_count)) {
    r <- as.integer(cn_count[i,3])
    M[,j:(j+r-1)] <- as.character(cn_count[i,1:2])
    j <- j + r
}

M <- t(M)

set.seed(4328904)
Sample_Size <- 50000 # number of samples, a parameter
M_s <- M[sample(nrow(M), Sample_Size, replace = TRUE),]
df_s <- setNames(as.data.frame(M_s), c("cause", "effect"))

# Next, get negative samples, ensuring that we dont have any legitimate pairs
set.seed(3192300)

matrix(sample(vocab, 100000 * 2, replace = TRUE), nrow = 100000, ncol = 2) %>%
    as.data.frame %>%
    setNames(c("cause", "effect")) %>%
    anti_join(cn_count, by=c("cause", "effect")) %>%
    sample_n(50000) ->
    df_n

# Combine
df_eval <-
    rbind(
        mutate(df_s, causal_pair = 1),
        mutate(df_n, causal_pair = 0)
    )

count(df_eval, causal_pair)

# Perform Evaluation ###########################################################


# Baseline
actl <- df_eval$causal_pair
pred <- sample(c(0,1), length(actl), replace = TRUE)
print(eval(pred, actl))

# Evaluation Results
L <- readRDS("matrices/cn_ce_100_2.rds")

M_c <- L$M_c
M_e <- L$M_e

Vocab <- as.environment(setNames(as.list(seq(vocab)), vocab))

pcis <- unlist(map2(df_eval$cause, df_eval$effect, function(x, y) {
    i <- Vocab[[as.character(x)]]
    j <- Vocab[[as.character(y)]]
    C_i <- M_c[i]
    E_j <- M_e[j]
    as.numeric(C_i %*% E_j)
}))

df_score <- mutate(df_eval, pci=pcis)
log_reg <- glm(causal_pair ~ pci, family = "binomial", data = df_score)
pred <- as.integer(predict(log_reg) > 0)
actl <- df_score$causal_pair

print(eval(pred, actl))

#> print(eval(pred, actl))
#f1 precision  recall accuracy
#1 0.4957128 0.5012883 0.49026  0.50126

