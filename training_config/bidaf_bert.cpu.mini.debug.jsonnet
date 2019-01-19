{
  "dataset_reader": {
    "type": "squad",
    "token_indexers": {
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": "datasets/bert/uncased_L-12_H-768_A-12/vocab.txt",
          "do_lowercase": false,
          "use_starting_offsets": true
      },
      "token_characters": {
        "type": "characters",
        "character_tokenizer": {
          "byte_encoding": "utf-8"
        }
      }
    }
  },
  "train_data_path": "allennlp/tests/fixtures/data/squad.json",
  "validation_data_path": "allennlp/tests/fixtures/data/squad.json",
  "model": {
    "type": "bidaf",
    "text_field_embedder": {
      "allow_unmatched_keys": true,
      "embedder_to_indexer_map": {
        "bert": ["bert", "bert-offsets"],
        "token_characters": ["token_characters"],
      },
      "token_embedders": {
        "bert": {
          "type": "bert-pretrained",
          "pretrained_model": "bert-base-uncased",
          "requires_grad": false,
          "top_layer_only": false
        },
        "token_characters": {
          "type": "character_encoding",
          "embedding": {
            "num_embeddings": 260,
            "embedding_dim": 8
          },
          "encoder": {
            "type": "cnn",
            "embedding_dim": 8,
            "num_filters": 8,
            "ngram_filter_sizes": [5]
          }
        }
      }
    },
    "num_highway_layers": 1,
    "phrase_layer": {
      "type": "lstm",
      "input_size": 776,
      "hidden_size": 776,
      "num_layers": 1
    },
    "similarity_function": {
      "type": "linear",
      "combination": "x,y,x*y",
      "tensor_1_dim": 776,
      "tensor_2_dim": 776
    },
    "modeling_layer": {
      "type": "lstm",
      "input_size": 3104,
      "hidden_size": 776,
      "num_layers": 1
    },
    "span_end_encoder": {
      "type": "lstm",
      "input_size": 5432,
      "hidden_size": 776,
      "num_layers": 1
    }
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["passage", "num_tokens"], ["question", "num_tokens"]],
    "padding_noise": 0.0,
    "batch_size": 40
  },
  "trainer": {
    "num_epochs": 1,
    "grad_norm": 10,
    "patience" : 12,
    "cuda_device" : -1,
    "optimizer": {
      "type": "adadelta",
      "lr": 0.5,
      "rho": 0.95
    }
  }
}
