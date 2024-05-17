package com.example.lottotest

data class ApiDataResponse(
    val cnn_prediction : String,
    val mlp_prediction : String,
    val rf_prediction : String,
    val rnn_prediction : String,
    val trans_prediction : String
)
